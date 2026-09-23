# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Author    : Luka Slagter
# Goal      : Data pipeline for level 2 JWST observations of NGC 346 (proposal #1227)

# to do     : filenameclass with easier use, only idx and filter input!
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

import os
os.chdir('/home/slagter/Astronomy_master/STRW-MSC/SF_in_NGC')

os.environ['CRDS_PATH'] = os.path.expanduser('~/crds_cache')
os.environ['CRDS_SERVER_URL'] = 'https://jwst-crds.stsci.edu'

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
from scipy.spatial import KDTree
from matplotlib.patches import Circle
from astropy.visualization import ZScaleInterval
from astropy.io import fits
from pathlib import Path
from matplotlib.colors import LogNorm
from astropy import wcs
from astropy.wcs.utils import proj_plane_pixel_scales
# 1/f calibration
from scripts.image1overf import sub1fimaging
import subprocess

# CRDS pathing etc.  
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits
from astropy.wcs.utils import skycoord_to_pixel

from astropy.nddata import extract_array
from astropy.visualization import (simple_norm,LinearStretch)
import jhat
import scipy.signal

wcs_align = jhat.st_wcs_align()
verbose=2
wcs_align.verbose=verbose

new_line = '\n'


class Read_Catalogue():
    def __init__(self, catalogue_path='Photometry/NGC346_NIRCam_and_MIRI_Public_JWST_Catalogue.fits'):
        self.catalogue_path = catalogue_path
        self.catalogue      = fits.open(catalogue_path)
        self.cat_data      = self.catalogue[1].data
        return

    def show_headers(self):
        with fits.open(self.catalogue_path) as hdulist:
            # Loop door alle beschikbare HDU-extensies
            for i, hdu in enumerate(hdulist):
                print(f"--- HEADER EXTENSIE {i} ({hdu.name}) ---")
                # repr() zorgt ervoor dat de kaarten netjes met regelafbrekingen worden getoond
                print(repr(hdu.header))
                print("\n" + "="*40 + "\n")
        return

    def mask_region(self, RA, DEC):
        self.cat_mask       = ((self.cat_data['RA'] <=RA[1]) & (self.cat_data['RA'] >= RA[0])) & ((self.cat_data['DEC'] <= DEC[1]) & (self.cat_data['DEC'] >= DEC[0]))
        self.sub_cat_data   = self.cat_data[self.cat_mask]
        return 



class Pipeline_level_2_data():
    """
    Helper class build around the in/output of starbug2
    """
    def __init__(self, working_directory=''):
        self.wdir = working_directory
        self.files = File_Finder(self.wdir)
        self.files.find_calibrated_filenames()
        self.IMAGE_TYPES = ('bgsub', 'cal', '1overf', 'background')

        print('Initialized data in', self.files.current_filters)

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    # (1) Full pipeline
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-         
    
    def run_pipeline_all(self, detection_sigma         = 2.0,
                                overwrite_always_run    = False,
                                save                    = False):
            
            filt1 = 'F115W'
            filt2 = 'F187N'
                
            for filt in [filt1, filt2]:
                print(f'Running for {filt}...')

                for IMAGE_INDEX in tqdm.tqdm(range(len(self.files.file_paths_1overf[filt])), desc='(1) Calibrating 1/f'): #
                    if Path(self.files.file_paths_1overf[filt][IMAGE_INDEX]).exists() == False:
                        self.calibrate_one_over_f(IMAGE_INDEX, filt)

                for IMAGE_INDEX in tqdm.tqdm(range(len(self.files.file_paths_1overf_jhat[filt])), desc='(2) Aligning with gaia catalog'): 
                    if Path(self.files.file_paths_1overf_jhat[filt][IMAGE_INDEX]).exists() == False:           
                        try:
                            self.assign_WCS_from_gaia_dr3(IMAGE_INDEX=IMAGE_INDEX, filter=filt)
                        except AttributeError:
                            continue
                            
                    
    def run_pipeline_single(self,  IMAGE_INDEX             = 28,
                            remove_background       = True,
                            detection_sigma         = 2.0,
                            overwrite_always_run    = False,
                            save                    = False):
        
        filt1 = 'F115W'
        filt2 = 'F187N'

        self.aperture_data  = {}
        self.wcs_align      = {}
        self.pixel_coord_sources = {}
        self.world_coord_sources = {}
        self.magnitudes     = {}
            
        for filt in [filt1, filt2]:
            print(f'Running for {filt}...')

            if Path(self.files.file_paths_1overf[filt][IMAGE_INDEX]).exists() == False | overwrite_always_run == True:
                print(f'    (1) Calibrating 1/f')
                self.calibrate_one_over_f(IMAGE_INDEX, filt)

            
            if Path(self.files.file_paths_1overf_jhat[filt][IMAGE_INDEX]).exists() == False| overwrite_always_run == True:
                print(f'    (2) Aligning with gaia catalog')
                self.assign_WCS_from_gaia_dr3(IMAGE_INDEX=IMAGE_INDEX, filter=filt)

            print(f'    (3) Running starbug')
            self.starbug2_aperture_photometry(  IMAGE_INDEX             = IMAGE_INDEX,
                                                filter                  = filt,
                                                remove_background       = True,
                                                toggle_sb_output_txt    = True,
                                                overwrite_always_run    = overwrite_always_run)

            self.aperture_data[filt]  = fits.open(self.files.file_paths_1overf_ap[filt][IMAGE_INDEX])[1].data
            
            # Use the gaia wcs of the image 
            self.wcs_align[filt]     = self._collect_wcs_of_SB2_detections(IMAGE_INDEX=IMAGE_INDEX,
                                            filter=filt,
                                            )
            # Pixel coordinates of detected sources
            self.pixel_coord_sources[filt], self.magnitudes[filt]    = self._collect_detections_mag_limited(filename_ap = self.files.file_paths_1overf_ap[filt][IMAGE_INDEX], filter=filt, detection_sigma= detection_sigma)
            self.world_coord_sources[filt]  = self._transform_detection_pixel_to_world(wcs= self.wcs_align[filt],
                                                                                        pixel_coords=self.pixel_coord_sources[filt] )

            if save == True:
                self.plot_cal_steps_per_filter(IMAGE_INDEX=IMAGE_INDEX, filter=filt, save = save)


        #_, _ , valid1, valid2   = self.cross_match_detections(self.world_coord_sources[filt1], self.world_coord_sources[filt2], det_tol=1e-5)
        # filters = [filt1, filt2]
        # validations = [valid1, valid2]
        # self.magnitudes = {}
        # for filt, valid in zip(filters, validations):
  
        #     self.magnitudes[filt] = self.aperture_data[filt][filt][valid]

        print('----Pipeline-----')
        self.find_calibration_versions(self.files.file_paths_1overf[filt][IMAGE_INDEX])
        print('----Completed----')
        return

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    # (2) 1/f calibration
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    def find_calibration_versions(self, filename):
        """
        print the used pipeline versions
        """
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
            print(prt + str(fits.open(filename)[0].header[f'{ext}']))
        return

    
    def calibrate_one_over_f(self, IMAGE_INDEX, filt):
        """
        Remove 1/f (pink noise) from  level 2 calibrated JWST observations using a row-wise median approach by using the
        image1overf.py tool from C. Willott (2022).

        Parameters are default
        """
        with fits.open(self.files.file_paths[filt][IMAGE_INDEX]) as cal2hdulist:
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

            cal2hdulist.writeto(self.files.file_paths_1overf[filt][IMAGE_INDEX], overwrite=True)
        return 

    
    def assign_WCS_from_gaia_dr3(self, IMAGE_INDEX, filter='F187N'):
        """
        Assing a World Coordinate System (wcs) to a single jwst field using the downloaded gaia catalog (5' around ngc 346)

        Uses the same sharpness/roundness parameters as used later by starbug, different per filter!!
        """
        out_put_dir = str(Path(self.files.file_paths[filter][IMAGE_INDEX]).parent.absolute().resolve())
        prev_cwd = os.getcwd()
        os.chdir(out_put_dir)
        config = load_starbug_params(self._determine_filename_sb2_param(filter=filter))
       
        gaia_cat_path =  str(Path(self.wdir).parent.parent) + '/ngc346_gaia_dr3_5arcmin_clean.txt'

        # with open(gaia_cat_path) as f:
        #     for i, line in enumerate(f):
        #         print(line)
        #         if i >= 5:
        #             break
        try:
            wcs_align.run_all(self.files.file_paths_1overf[filter][IMAGE_INDEX],
                    telescope= 'jwst',
                    outsubdir= out_put_dir ,
                    imagetype= 'cal',                                    #Need to specify as the filename is ext3
                    overwrite=True,
                    d2d_max=.5,
                    showplots=0,
                    refcatname=gaia_cat_path,
                    refcat_racol='ra',
                    refcat_deccol='dec',
                    refcat_magcol='phot_g_mean_mag',
                    refcat_magerrcol='phot_g_mean_mag_error',
                    refcat_pmflag=False,
                    histocut_order='dxdy',
                        sharpness_lim=(config.get('SHARP_LO'),config.get('SHARP_HI')),
                        roundness1_lim=(config.get('ROUND_LO'), config.get('ROUND_HI')),
                        SNR_min= config.get('SIGSRC'),
                        dmag_max=1.0,
                        objmag_lim =(14, 24))
        #regardless of the try above go back to previous cwd
        finally:
            os.chdir(prev_cwd)
            
        return 

    def _read_fits_detections(self, filename_ap, filter):   
        """
        Collects the x and y position alongside the magnitude error of detected sources from a starbug2 aperture photometry result file.
        """                                                                 
        x_positions_sources = fits.open(filename_ap)[1].data[f'xcentroid']
        y_positions_sources = fits.open(filename_ap)[1].data[f'ycentroid']
        magnitude_sources = fits.open(filename_ap)[1].data[filter]
        magnitude_error_sources = fits.open(filename_ap)[1].data['e'+ filter]
        return x_positions_sources, y_positions_sources, magnitude_sources,  magnitude_error_sources

    def _collect_detections_mag_limited(self, filename_ap, filter, detection_sigma = 2.0):
        """
        Provides the x and y-pixel positions of all detected sources from a starbug2 aperture file.
        Additionally outputs mask filtering which detections are below the provided maximal error in magnitude
        """
        x_positions_sources, y_positions_sources, magnitude,  magnitude_error_sources = self._read_fits_detections(filename_ap = filename_ap, filter= filter)
        detection_mask              = magnitude_error_sources <= detection_sigma
        return (x_positions_sources[detection_mask], y_positions_sources[detection_mask]), magnitude[detection_mask]

    def _transform_detection_pixel_to_world(self, wcs, pixel_coords):
        """
        Transform PIXEL to WORLD coordinates with provided wcs. pixel_coords must be (x_pos, y_pos)
        """
        return wcs.pixel_to_world(*pixel_coords) 

    def _transform_detection_world_to_pixel(self, wcs, world_coords):
        """
        Transform WORLD to PIXEL coordinates with provided wcs. world_coords must be (ra, dec)
        """
        return wcs.world_to_pixel_values(*world_coords) 
    
    def _collect_wcs_of_SB2_detections(self, IMAGE_INDEX=0, filter='F187N'):
        """
        Find the wcs system from the gaia matchinging
        """
        align_fits = fits.open(self.files.file_paths_1overf_jhat[filter][IMAGE_INDEX])
        return  wcs.WCS(align_fits['SCI', 1], align_fits)

    def cross_match_detections(self, sky1, sky2, det_tol=0.5 * u.arcsec):
        """
        Build a kd-tree from the detection coordinates of filter [A], and find all coordinates within 0.5'' of it in filter [B].
        """
        A = np.array([sky1.ra, sky1.dec]).T
        B = np.array([sky2.ra, sky2.dec]).T
   
        tree    = KDTree(B)
        matches = tree.query_ball_point(A, r = det_tol)

        a_idx = []
        b_idx = []
        for i, match in enumerate(matches):
            for j in match:
                a_idx.append(i)
                b_idx.append(j)

        a_idx = np.array(a_idx)
        b_idx = np.array(b_idx)
        print(f'Found {len(a_idx)} matching detections')
        return A, B, a_idx, b_idx


    def _find_overlapping_indices(self, filters, det_tol):
        """
        Search for overlapping sources between filters. method:
        1) set first filter as reference
        2) find all detections that occur in filter B within det_tol radius of each detecion in A
        3) select only detections that occur in both filters
        """
        ref = filters[0]
        ref_idx = np.arange(len(self.world_coord_sources[ref][0]))
        matched_mask = np.ones(len(ref_idx), dtype=bool)

        pair_matches = {}  # other_filter -> (a_idx into ref, b_idx into other)
        for other in filters[1:]:
            _, _, a_idx, b_idx = self.cross_match_detections(
                sky1 = self.world_coord_sources[ref], sky2 = self.world_coord_sources[other], det_tol=det_tol
            )
            pair_matches[other] = (a_idx, b_idx)

            has_match = np.zeros(len(ref_idx), dtype=bool)
            if len(a_idx) > 0:
                has_match[np.unique(a_idx)] = True
            matched_mask &= has_match

        overlap_ref_idx        = ref_idx[matched_mask]
        overlap_idx_per_filter = {ref: overlap_ref_idx}
        for other in filters[1:]:
            a_idx, b_idx = pair_matches[other]
            if len(a_idx) > 0:
                keep = np.isin(a_idx, overlap_ref_idx)
                overlap_idx_per_filter[other] = np.unique(b_idx[keep])
            else:
                overlap_idx_per_filter[other] = np.array([], dtype=int)

        return overlap_idx_per_filter

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    # (3) Aperture photometry / source detection with STARBUG2
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    def _run_starbug2(self, cmd, toggle_sb_output_txt):
        """
        Performs starbug2 aperture photometry using subproccess
        """
        print("Launching Starbug environment in its venv...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("STDOUT:\n", result.stdout)
            print("STDERR:\n", result.stderr)
            result.check_returncode()

        if toggle_sb_output_txt == True:
            print(result.stdout)
            print("---------------------------")
            print()
        return

    def _determine_filename_sb2_param(self, filter):
        return str('/home/slagter/Astronomy_master/STRW-MSC/SF_in_NGC/Photometry/parameter_files_starbug/starbug_' + filter + ".param")

    def starbug2_aperture_photometry(self, IMAGE_INDEX, filter, remove_background=True, toggle_sb_output_txt=True, overwrite_always_run=False):
        """
        Wrapper function with standard file paths and cmd lines to run aperture photomety starbug2 using subprocceses. 
        
        only does aperture photometry if results are not yet present or the overwrite always.is true

        parameter file should exist, and have to correct naming convention for each filter!
        """
        starbug_path = "/net/vdesk/data2/slagter/environment_path/SB_venv/bin/starbug2"
        mode = "-vBD" if remove_background else "-vD"

        cmd = [
            starbug_path,
            mode,
            "-p", self._determine_filename_sb2_param(filter),
            self.files.file_paths_1overf[filter][IMAGE_INDEX],
        ]

        if overwrite_always_run:
            self._run_starbug2(cmd, toggle_sb_output_txt)
        else:
            ap_exists = Path(self.files.file_paths_1overf_ap[filter][IMAGE_INDEX]).exists()
            bgd_exists = Path(self.files.file_paths_1overf_bgd[filter][IMAGE_INDEX]).exists()

            if ap_exists and bgd_exists:
                print("Skipping starbug2 run")
            elif ap_exists and remove_background == False:
                print("Skipping starbug2 run")
            else:
                self._run_starbug2(cmd, toggle_sb_output_txt)
    
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    # (4) Plotting functions
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    def _collect_image(self, filename):
        return fits.open(filename)[1].data

    def plot_cal_steps_per_filter(self, IMAGE_INDEX, filter, save= False):
        print("Plotting Level 2 downloaded data...")

        fig = plt.figure(figsize=(30, 7))
        gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.0, hspace=0.0)
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[0, 2])
        ax3 = fig.add_subplot(gs[0, 3])

        image   = self._collect_image(self.files.file_paths[filter][IMAGE_INDEX] )

        interval = ZScaleInterval()
        vmin, vmax = interval.get_limits(image)

        ax0.imshow(image, cmap='gray',  vmin=vmin, vmax=vmax, origin='lower')

        oneoverf   = self._collect_image(self.files.file_paths_1overf[filter][IMAGE_INDEX] )
        ax1.imshow(oneoverf, cmap='gray',  vmin=vmin, vmax=vmax, origin='lower')


        bgd   = self._collect_image(self.files.file_paths_1overf_bgd[filter][IMAGE_INDEX])
        ax2.imshow(image, cmap='gray',  vmin=vmin, vmax=vmax, origin='lower')

        oneoverf_sub_bgd = oneoverf- bgd
        ax3.imshow(oneoverf_sub_bgd, cmap='gray',  vmin=vmin, vmax=vmax, origin='lower')

        fontsize = 26
        ax0.set_title('Level 2 downloaded data' , fontsize=fontsize)
        ax1.set_title('1/f calibrated'          , fontsize=fontsize)
        ax2.set_title('background'              , fontsize=fontsize)
        ax3.set_title('background calibrated'   , fontsize=fontsize)

        for ax in [ax0,ax1,ax2,ax3]:
            ax.set_yticklabels([])
            ax.set_xticklabels([])
        fig.suptitle(filter, fontsize=fontsize+10)

        if save == True:
            plt.savefig(f'Plots/bgd_cal_steps_{IMAGE_INDEX}_{filter}.pdf', bbox_inches='tight')
        plt.show()

     
    def show_detections(self,
                        filters=None,
                        image_type='bgsub',
                        IMAGE_INDEX=28,
                        draw_detections=True,
                        specific_ra_dec = None,
                        marker_radius_pixel=3,
                        save=True,
                        save_dir='Plots'):
        """
        Visualize STARBUG2 detections (and/or user-supplied markers) for one or more filters.
    
        Parameters
        ----------
        filters : 
            Filter to plot
        image_type : {'bgsub', 'cal', '1overf', 'background'}
            Which image to show underneath (default: background-subtracted).
            'bgsub'      : 1/f-calibrated minus background
            'cal'        : raw level 2 (cal) image
            '1overf'     : 1/f-calibrated image
            'background' : background model only

        """
        # check thingy
        if image_type not in self.IMAGE_TYPES:
            raise ValueError(f"Unknown image : '{image_type}'. Use one of {self.IMAGE_TYPES}.")

        for filt in filters:
            if filt not in self.wcs_align:
                raise KeyError(
                    f"'{filt}' has no WCS yet - run run_pipeline()...) first! "
                )
    
            def _load_image(filt):
                f = self.files
                if image_type == 'cal':
                    img = self._collect_image(f.file_paths[filt][IMAGE_INDEX])
                elif image_type == '1overf':
                    img = self._collect_image(f.file_paths_1overf[filt][IMAGE_INDEX])
                elif image_type == 'background':
                    img = self._collect_image(f.file_paths_1overf_bgd[filt][IMAGE_INDEX])
                else:  # 'bgsub'
                    img = (self._collect_image(f.file_paths_1overf[filt][IMAGE_INDEX])
                        - self._collect_image(f.file_paths_1overf_bgd[filt][IMAGE_INDEX]))
                vmin, vmax = ZScaleInterval().get_limits(img)
                return img, vmin, vmax
            
        
            fig = plt.figure()
            gs = gridspec.GridSpec(1, 1, figure=fig)   # once, before the loop

            ax = fig.add_subplot(gs[0,0], projection=self.wcs_align[filt]) 
                
            img, vmin, vmax = _load_image(filt)
            ax.imshow(img, cmap='bone', vmin=vmin, vmax=vmax, origin='lower')

            from matplotlib.collections import EllipseCollection

            def _draw_markers(ax, x, y, r, img_shape, label):
                """Detections; points are dropped if not in frame"""
                ok = (x >= 0) & (x < img_shape[1]) & (y >= 0) & (y < img_shape[0])

                ax.add_collection(EllipseCollection(
                    2*r, 2*r, 0, units='xy', offsets=np.c_[x[ok], y[ok]],
                    offset_transform=ax.transData,   # transOffset= on matplotlib < 3.6
                    fc='none', ec='magenta', lw=0.5, label=label))
    
            if draw_detections:
                x, y = self.pixel_coord_sources[filt]
                _draw_markers(ax, x, y, marker_radius_pixel, img.shape, 'detections')

            if specific_ra_dec is not None:
                x, y = self.wcs_align[filt].world_to_pixel_values(*specific_ra_dec)
                _draw_markers(ax, x, y, marker_radius_pixel, img.shape, 'catalogue')
                            

            ax.coords['dec'].set_ticks(spacing=50 * u.arcsec)
            ax.coords['ra'].set_ticks(spacing=50 * u.arcsec)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_title(filt)
            ax.legend(loc='upper right')
        
            title = {'bgsub': 'background-subtracted', 'cal': 'level 2 (cal)',
                    '1overf': '1/f calibrated', 'background': 'background only'}[image_type]
            plt.suptitle(title)
            fig.tight_layout()
        
            if save:
                os.makedirs(save_dir, exist_ok=True)
                tag = 'Starbug2' if draw_detections else 'Jeroen' if specific_ra_dec is not None else 'pure'

                fig.savefig(f'{save_dir}/detections_{image_type}_{tag}.pdf', bbox_inches='tight')
        
            plt.show()
        return




class File_Finder:
    """
    Finds calibrated (_cal.fits) files under a working directory and
    splits them by filter, read from the FITS header FILTER keyword.
    """

    def __init__(self, wdir, filter_patterns=None):
        """
        wdir: directory containing one subfolder per exposure
              (e.g. mastDownload/JWST)
        filter_patterns: optional dict mapping filter name -> list of
              folder-name substrings, used only as a fallback if a
              file's header can't be read.
        """
        self.wdir = Path(wdir)
        self.file_paths = {}
        self.unmatched = []

    def _filter_from_header(self, cal_path):
        """
        Obtain the filter in a given .fits file if possible
        """
        try:
            return fits.getval(cal_path, 'FILTER', ext=0)
        except Exception as e:
            print(f"Warning: couldn't read FILTER from {cal_path.name}: {e}")
            return None

    def _change_extentions(self, input_cal_file, mode='1overf'):
        """
        Transform filename to several calibration steps 
        """
        if mode == '1overf':
            name = input_cal_file.replace('_cal.fits','_cal_1overf_cal.fits')
        elif mode == '1overf_bg':
            name = input_cal_file.replace('_cal.fits','_cal_1overf_cal-bgd.fits')
        elif mode == '1overf_ap':
            name = input_cal_file.replace('_cal.fits','_cal_1overf_cal-ap.fits')
        elif mode == 'jhat':
            name = input_cal_file.replace('_cal.fits','_cal_1overf_jhat.fits')

        else:
            print(f'{mode} is not valid')
            return
        return name
        
    def find_calibrated_filenames(self):
        """
        Read the directory for all files. read the filter from the fits file and store appropriately per filter.
        """

        self.current_filters = []
        for dir_path in sorted(p for p in self.wdir.iterdir() if p.is_dir()):
            cal_path = dir_path / f"{dir_path.name}_cal.fits"
            if not cal_path.exists():
                continue

            filt = self._filter_from_header(cal_path)
            if filt not in self.current_filters:
                self.current_filters.append(filt)

            if filt is None:
                self.unmatched.append(str(cal_path))
                continue

            self.file_paths.setdefault(filt, []).append(str(cal_path))

        for mode, attr in [
            ('1overf',      'file_paths_1overf'),
            ('1overf_ap',   'file_paths_1overf_ap'),
            ('1overf_bg',   'file_paths_1overf_bgd'),
            ('jhat',        'file_paths_1overf_jhat'),
        ]:
            setattr(self, attr, {
                f: [self._change_extentions(name, mode=mode) for name in names]
                for f, names in self.file_paths.items()
            })

        if self.unmatched:
            print(f"Warning: {len(self.unmatched)} files had no resolvable filter")

        return 



class GaussianPointSource:
    """Generates a mock image containing a single 2D Gaussian point
    source on top of a flat sky background, with optional Poisson shot
    noise.
    """

    def __init__(self, shape=(101, 101), amplitude=1000.0, x0=None,
                 y0=None, sigma=3.0, background=50.0, gain=2.0,
                seed=42):

        self.shape      = shape
        self.amplitude  = amplitude
        self.x0         = shape[1] / 2.0 if x0 is None else x0
        self.y0         = shape[0] / 2.0 if y0 is None else y0
        self.sigma      = sigma
        self.background = background
        self.gain       = gain
        self.rng        = np.random.default_rng(seed)

    def true_flux(self):
        """
        Analytic total flux of the Gaussian source
        """
        return self.amplitude * 2.0 * np.pi * self.sigma ** 2

    def encircled_energy(self, radius):
        return 1.0 - np.exp(-radius ** 2 / (2.0 * self.sigma ** 2))

    def _gaussian_signal(self):
        ny, nx  = self.shape
        y, x    = np.mgrid[0:ny, 0:nx]
        r2      = (x - self.x0[:, np.newaxis, np.newaxis]) ** 2 + (y - self.y0[:, np.newaxis, np.newaxis]) ** 2
        return np.sum(self.amplitude * np.exp(-r2 / (2.0 * self.sigma ** 2)), axis=0)

    def _bgd_noise(self):

        # Source - https://stackoverflow.com/a/63868276
        # Posted by Igor
        # Retrieved 2026-09-11, License - CC BY-SA 4.0

        # Compute filter kernel with radius correlation_scale (can probably be a bit smaller)
        correlation_scale = self.shape[0]/2
        x = np.arange(-correlation_scale, correlation_scale)
        y = np.arange(-correlation_scale, correlation_scale)
        X, Y = np.meshgrid(x, y)
        dist = np.sqrt(X*X + Y*Y)
        filter_kernel = np.exp(-dist**2/(2*correlation_scale))

        n       =self.shape[0]
        noise = np.random.randn(n, n)
        noise = scipy.signal.fftconvolve(noise, filter_kernel, mode='same')
        return self.background * ((noise - noise.min())/ (noise.max() - noise.min()) )


    def clean_image(self):
        """
        Noiseless image
        """
        return self._gaussian_signal() + self.background

    def noisy_image(self):
        """
        simple noise model:
        Poisson shot noise on the (source + background) 
        """
        clean_adu           = self.clean_image()

        electrons           = clean_adu * self.gain
        noisy_electrons     = self.rng.poisson(electrons).astype(float)
        noisy_bgd           = self._bgd_noise()
        return noisy_electrons / self.gain + noisy_bgd

    def save_fits(self, filename, noisy=False, overwrite=True):
        """Write the image to a FITS file, with the
        ground-truth parameters recorded in the header for reference.
        """
        data                = self.noisy_image() if noisy else self.clean_image()
        hdu                 = fits.PrimaryHDU(data.astype(np.float32))
        hdr                 = hdu.header
        hdr["SIGMA"]        = (self.sigma, "Gaussian sigma [pix]")
        hdr["AMP"]          = (self.amplitude, "Gaussian peak amplitude [ADU]")
        # hdr["XTRUE"]        = (self.x0, "true x centroid [pix]")
        # hdr["YTRUE"]        = (self.y0, "true y centroid [pix]")
        hdr["BKG"]          = (self.background, "flat sky background [ADU]")
        hdr["GAIN"]         = (self.gain, "e-/ADU")
        hdr["TRUFLUX"]      = (self.true_flux(), "analytic total source flux [ADU]")
        hdr["NOISY"]        = (noisy, "was noise injected?")

        hdu.writeto(filename, overwrite=overwrite)
        return Path(filename)

    def write_source_list(self, filename, overwrite=True):
        """
        Write a minimal binary-table FITS source list containing
        just the one known-true centroid, in the xcentroid/ycentroid
        column format starbug2's aperture photometry routine expects.
        """
        t = Table()
        t["xcentroid"] = list(self.x0)
        t["ycentroid"] = list(self.y0)
        t.write(filename, format="fits", overwrite=overwrite)
        return Path(filename)

    def plot_cal_steps(self, noisy_fits, bgd_fits=None, ap_table=None,
                    apphot_r=None, sky_rin=None, sky_rout=None,
                    save=False, save_dir='Plots'):

        fig = plt.figure(figsize=(21, 7))
        gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.0, hspace=0.0)
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[0, 2])

        cmap = 'inferno'
        noisy = fits.open(noisy_fits)[0].data 

        interval   = ZScaleInterval()
        vmin, vmax = interval.get_limits(noisy)

        ax0.imshow(noisy, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')


        bgd = fits.open(bgd_fits)[1].data 

        ax1.imshow(bgd, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')

        subtracted = noisy - bgd
        ax2.imshow(subtracted, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')

        fontsize = 26
        ax0.set_title('Mock (noisey))',       fontsize=fontsize)
        ax1.set_title('Background only (sb2 result)',   fontsize=fontsize)
        ax2.set_title('Background calibrated (Mock - Bgd)',   fontsize=fontsize)

        for ax in [ax0, ax1, ax2]:
            ax.set_yticklabels([])
            ax.set_xticklabels([])

        if save:
            Path(save_dir).mkdir(exist_ok=True)
            plt.savefig(f'{save_dir}/gaussian_cal_steps.pdf', bbox_inches='tight')
        plt.show()
        return 



def load_starbug_params(file_path):
    """
    READS StarBugII .param file
    """
    params = {}
    
    with open(file_path, 'r') as file:
        for line in file:
            clean_line = line.strip()
            
            if not clean_line or clean_line.startswith('//'):
                continue
            
            # Starbug2 : 'key = value // comment'
            if '=' in clean_line:
                key_part, rest = clean_line.split('=', 1)
                key = key_part.strip()
                
                # Remoce inline comments
                value_part = rest.split('//')[0].strip()
                
                # Datatype convertion (float, int of string)
                try:
                    if '.' in value_part:
                        value = float(value_part)
                    else:
                        value = int(value_part)
                except ValueError:
                    value = value_part 
                
                params[key] = value
                
    return params

