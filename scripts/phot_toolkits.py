


import numpy as np, matplotlib.pyplot as plt, matplotlib.gridspec as gridspec, astropy.units as u, astropy.constants as const, os, sys

import tqdm
plt.style.use('/home/slagter/Astronomy_master/STRW-MSC/stylish.mplstyle')
from astropy.table import Table
from astropy.table import Table
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation
import matplotlib.ticker as ticker
from matplotlib.ticker import FixedLocator, AutoMinorLocator, MultipleLocator, LogLocator, NullFormatter
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from IPython import display

from astropy.visualization import ZScaleInterval
from astropy.io import fits
import os
from pathlib import Path
from matplotlib.colors import LogNorm

from scripts.image1overf import sub1fimaging


import subprocess
import photutils

import jhat
from jhat import jwst_photclass, st_wcs_align
wcs_align       = st_wcs_align()
# wcs_align_batch = align_wcs_batch()


new_line = '\n'
class Pipeline_level_2_data():
    """
    Helper class build around the in/output of starbug2
    """
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    # (1) Full pipeline
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    def __init__(self, working_directory='', filter=''):
        self.filter = filter
        self.wdir   = working_directory + self.filter + '/mastDownload/JWST/'
        pass

    def run_pipeline(self):
        self._find_calibrated_filenames()
        self.N_fields               = len(self.cal_file_names)

        print(f'We have {self.N_fields} fields!')
        self.calibrate_one_over_f_total(self)

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    # (2) 1/f calibration
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    def find_calibration_versions(self, filename):

        proccessing_ext     = [ 'CAL_VER',
                                'CRDS_VER', 
                                'CRDS_CTX',
                                'READPATT',
                                'NGROUPS']
        
        proccessing_print   = ['JWST pipeline version             : ',
                               'Calibration Reference Data System : ',
                               'CRDS context map                  : ',
                               'Read-out pattern                  : ',
                               'Number of read-out groups         : '] 
        
        for prt, ext in zip(proccessing_print, proccessing_ext):
            print(prt + fits.open(filename)[0].header[f'{ext}'])
        return


    def calibrate_one_over_f_total(self):
        for filename in tqdm.tqdm(self.cal_file_names, desc= '1/f correction (C. Willott 2022)'):
            cal21overffile      = self._change_extentions(input_cal_file=filename,
                                                          mode='1overf')
            if Path(cal21overffile).exists() == False:
                self._calibrate_one_over_f(filename=filename)
        
    
    def _calibrate_one_over_f(self, filename):
        cal21overffile  = self._change_extentions(input_cal_file = filename,
                                                 mode           ='1overf')
        with fits.open(filename) as cal2hdulist:
            if cal2hdulist['PRIMARY'].header['SUBARRAY']=='FULL' or cal2hdulist['PRIMARY'].header['SUBARRAY']=='SUB256':
                sigma_bgmask    = 3.0
                sigma_1fmask    = 2.0
                splitamps       = False     #Set to True only in a sparse field so each amplifier will be fit separately. 
                usesegmask      = True
                correcteddata   = sub1fimaging(cal2hdulist,sigma_bgmask,sigma_1fmask,splitamps,usesegmask)

            if cal2hdulist['PRIMARY'].header['SUBARRAY']=='FULL':
                cal2hdulist['SCI'].data[4:2044,4:2044] = correcteddata  
            elif cal2hdulist['PRIMARY'].header['SUBARRAY']=='SUB256':
                cal2hdulist['SCI'].data[:252,:252] = correcteddata

            cal2hdulist.writeto(cal21overffile, overwrite=True)

        return cal21overffile

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    # (3) Aperture photometry / source detection with STARBUG2
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    def _run_starbug2(self, param_cmd, cmd, toggle_sb_output_txt):
        print("Launching Starbug environment in its venv...")
        subprocess.run(param_cmd)
        print('Changed to correct param file')
        result = subprocess.run(cmd,
                                capture_output=toggle_sb_output_txt,    # Save terminal output 
                                text=toggle_sb_output_txt,              # Convert output to string text
                                check=True                              # Crash main script if starbug fails
                            )

        if toggle_sb_output_txt == True:
            print(result.stdout)
            print("---------------------------")
            print()
        return


    def starbug2_aperture_photometry(self, filename, sb2_param_path, remove_background = True, toggle_sb_output_txt = True, overwrite_always_run=False):
        '''
        TO DO: 
            - [] Add different parameter files based on filter

        '''
        starbug_path        = "/net/vdesk/data2/slagter/environment_path/SB_venv/bin/starbug2" 
 
        if remove_background == True:
            mode = "-vBD"
        else:
            mode = "-vD"

        param_cmd   =  [starbug_path, "-p", sb2_param_path]
        cmd         = [starbug_path, mode, filename]

        if overwrite_always_run == True:
                self._run_starbug2(param_cmd, cmd, toggle_sb_output_txt)
        else:
            check_path_ap = Path(self._change_extentions(input_cal_file=filename,
                                                            mode='1overf_ap'))
            check_path_bg = Path(self._change_extentions(input_cal_file=filename,
                                                            mode='1overf_bg'))

                #Check so we dont rerun starbug2 runs already done
            if check_path_ap.exists() == True and check_path_bg.exists()  == True: 
                print("Skipping starbug2 run")

                #If we dont want background this check is enough
            elif check_path_ap.exists() == True and remove_background==False:
                print("Skipping starbug2 run")

                #Run if either Background subtraction wasnt done or no SB at all
            else:
                self._run_starbug2(param_cmd, cmd, toggle_sb_output_txt)



    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    # (3) Plotting functions
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    def show_one_over_f_SB2(self, filename, safefig_path = None, detection_sigma=1.0):

        one_over_f_cal_image_path = self._change_extentions(input_cal_file=filename,
                                             mode='1overf')
        SB2_det_path        = self._change_extentions(input_cal_file=filename,
                                             mode='1overf_ap')
        SB2_bgd_path        = self._change_extentions(input_cal_file=filename,
                                             mode='1overf_bg')

        one_over_f_cal_image    = self._collect_image(one_over_f_cal_image_path)
        SB2_bgd_image           = self._collect_image(SB2_bgd_path)
        
        interval = ZScaleInterval()
        vmin, vmax = interval.get_limits(one_over_f_cal_image)

        fig = plt.figure(figsize=(14, 14))
        gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.05, hspace=0.1)
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])

        ax0.imshow(one_over_f_cal_image                 , cmap='gray',  vmin=vmin, vmax=vmax, origin='lower')
        ax0.set_title('Im1 : Level 2 image after 1/f calibration')

        ax1.imshow(SB2_bgd_image                        , cmap='gray',  vmin=vmin, vmax=vmax, origin='lower')
        ax1.set_title('Im2 : Background')

        ax2.imshow(one_over_f_cal_image-SB2_bgd_image   , cmap='gray',  vmin=vmin, vmax=vmax, origin='lower')
        ax2.set_title('Im3 : Background removal')

        ax3.imshow(one_over_f_cal_image-SB2_bgd_image   , cmap='gray',  vmin=vmin, vmax=vmax, origin='lower')
        ax3.set_title('Im3 : SB2 source detection')

        x_positions_sources, y_positions_sources, magnitude_error_sources = self._read_fits_detections( filename_ap = SB2_det_path)

        detection_mask      = magnitude_error_sources <= detection_sigma

        # Overplot the source detections as open circles
        ax3.scatter(x_positions_sources[detection_mask], y_positions_sources[detection_mask], s=35, facecolors='none', edgecolors='red', linewidths=0.5, label=f'Starbug2 Detections sigma_mag < {detection_sigma}')
        ax3.scatter(x_positions_sources[~detection_mask], y_positions_sources[~detection_mask], s=35, facecolors='none', edgecolors='cyan', linewidths=0.5,  label=f'Starbug2 Detections sigma_mag > {detection_sigma}')
          

        ax0.set_yticklabels([])
        ax1.set_yticklabels([])
        ax2.set_yticklabels([])
        ax3.set_yticklabels([])
        ax0.set_xticklabels([])
        ax1.set_xticklabels([])
        ax2.set_xticklabels([])
        ax3.set_xticklabels([])
        ax3.legend()

        fig.suptitle(f'{filename.replace(f'../../../../../../../../net/vdesk/data2/slagter/JWST_NIRCAM*_#1227_level_2/{self.filter}/mastDownload/JWST/','')}' + new_line + self.filter )

        if safefig_path != None:
            plt.savefig(safefig_path, bbox_inches='tight' )
        plt.show()

    def show_level2_and_one_over_f_difference(self, filename):
        cal21overffile      = filename.replace('_cal.fits','_cal_1overf.fits')
        if Path(cal21overffile).exists() == False:
            self._calibrate_one_over_f(filename=filename)

        cal_image           = self._collect_image(filename)
        cal_one_over_f_image= self._collect_image(cal21overffile)
        
        interval = ZScaleInterval()
        vmin, vmax = interval.get_limits(cal_one_over_f_image)

        fig = plt.figure(figsize=(21, 7))
        gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.05, hspace=0.0)
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[0, 2])

        ax0.imshow(cal_image, cmap='gray',  vmin=vmin, vmax=vmax, origin='lower')
        ax0.set_title('Im1 : Level 2 image')

        ax1.imshow(cal_one_over_f_image, cmap='gray',  vmin=vmin, vmax=vmax, origin='lower')
        ax1.set_title('Im2 : Level 2 image after 1/f calibration')

        ax2.imshow(np.abs(cal_image-cal_one_over_f_image), cmap='gray',  vmin=vmin, vmax=vmax, origin='lower')
        ax2.set_title('∆Im = |Im1 - Im2|  ')


        ax0.set_yticklabels([])
        ax1.set_yticklabels([])
        ax2.set_yticklabels([])
        ax0.set_xticklabels([])
        ax1.set_xticklabels([])
        ax2.set_xticklabels([])
        fig.suptitle(f'{filename.replace(f'../../../../../../../../net/vdesk/data2/slagter/JWST_NIRCAM*_#1227_level_2/{self.filter}/mastDownload/JWST/','')}'+ new_line + self.filter ) 
        plt.show()

    def show_detections_over_field(self, file_index=0,show_detections=True):
        fig = plt.figure(figsize=(7, 7))
        gs = gridspec.GridSpec(1, 1, figure=fig, wspace=0.1, hspace=0.0)
        ax0 = fig.add_subplot(gs[0, 0])

        image       = self._collect_image(self.filenames_jwst_2c[file_index])

        interval = ZScaleInterval()
        vmin, vmax = interval.get_limits(image)
    
        ax0.imshow(image, cmap='gray',  vmin=vmin, vmax=vmax, label=f'{self.filter_per_field}')

        
        # Read the source detections in x, y coordinates (pixel coordinates)
        if show_detections == True:
            x_positions_sources, y_positions_sources = self._read_fits_collumn( filename      = self.filenames_sb2_ap_phot[file_index],
                                                                                target_data   = ['xcentroid', 'ycentroid'],
                                                                                col_num       = 1)
            # Overplot the source detections as open circles
            ax0.scatter(x_positions_sources, y_positions_sources, s=25, facecolors='none', edgecolors='red', linewidths=0.4, label='Starbug2 Detections')
            ax0.set_title(f'Starbug2 Source Detections : {new_line} {self.filenames[file_index]}', x=0.5, y=1.05)

        else:
            ax0.set_title(f'JWST Field : {new_line} {self.filenames[file_index]}', x=0.5, y=1.05)
            

        ax0.legend()
        
        
        ax0.set_xlabel(r'x [pixel]')
        ax0.set_ylabel(r'y [pixel]')
        
        plt.show()
        return

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    # (4) Helper functions for fits files
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    def _read_fits_collumn(self, filename,  target_data, col_num = 1, ):
        output = []
        for target_data_i in target_data:
            output.append(fits.open(filename)[col_num].data[f'{target_data_i}'])
        return output

    def _read_fits_detections(self, filename_ap):                                                                    
            x_positions_sources = fits.open(filename_ap)[1].data[f'xcentroid']
            y_positions_sources = fits.open(filename_ap)[1].data[f'ycentroid']
            magnitude_error_sources = fits.open(filename_ap)[1].data['e'+self.filter]
            return x_positions_sources, y_positions_sources, magnitude_error_sources

    def _collect_image(self, filename):
        return fits.open(filename)[1].data
    
    def _determine_filename_sb2_param(self,):
        param_path  = "/net/vdesk/data2/slagter/environment_path/SB_venv/lib/python3.11/site-packages/starbug2/"
        return param_path + str('param' + '_' + self.filter + ".py")

    def _find_calibrated_filenames(self):
        wdir = Path(self.wdir)
        self.dir_names = []
        self.cal_file_names = []

        for dir_path in sorted(p for p in wdir.iterdir() if p.is_dir()):
            cal_path = dir_path / f"{dir_path.name}_cal.fits"
            if cal_path.exists():
                self.dir_names.append(str(dir_path.name))
                self.cal_file_names.append(str(cal_path))
        return

    def _change_extentions(self, input_cal_file, mode='1overf'):
        if mode == '1overf':
            name = input_cal_file.replace('_cal.fits','_cal_1overf.fits')
        elif mode == '1overf_bg':
            name = input_cal_file.replace('_cal.fits','_cal_1overf-bgd.fits')
        elif mode == '1overf_ap':
            name = input_cal_file.replace('_cal.fits','_cal_1overf-ap.fits')
        else:
            print(f'{mode} is not valid')
            return
        return name


    



    



