# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Author    : Luka Slagter
# Goal      : Data pipeline for level 2 JWST observations of NGC 346 (proposal #1227)

# to do     : filenameclass with easier use, only idx and filter input!
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


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

from astropy.visualization import ZScaleInterval
from astropy.io import fits
from pathlib import Path
from matplotlib.colors import LogNorm

# 1/f calibration
from scripts.image1overf import sub1fimaging
import subprocess

# CRDS pathing etc.  
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits
from astropy.wcs.utils import skycoord_to_pixel
from astropy import wcs
from astropy.nddata import extract_array
from astropy.visualization import (simple_norm,LinearStretch)
import jhat
import scipy.signal

wcs_align = jhat.st_wcs_align()
verbose=2
wcs_align.verbose=verbose

new_line = '\n'


class Pipeline_level_2_data():
    """
    Helper class build around the in/output of starbug2
    """
    def __init__(self, working_directory=''):
        self.wdir = working_directory
        self.files = File_Finder(self.wdir)
        self.files.find_calibrated_filenames()

        print('Initialized data in', self.files.current_filters)

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    # (1) Full pipeline
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-         
    def run_pipeline(self,  IMAGE_INDEX             = 28,
                            remove_background       = True,
                            detection_sigma         = 2.0,
                            overwrite_always_run    = False,
                            save                    = False):
        

        filt1 = 'F115W'
        filt2 = 'F187N'

        self.aperture_data  = {}
        self.wcs_align      = {}
        self.sky            = {}
            
        for filt in [filt1, filt2]:
            print(f'Running for {filt}...')

            if Path(self.files.file_paths_1overf[filt][IMAGE_INDEX]).exists() == False:
                self._calibrate_one_over_f(IMAGE_INDEX, filt)

            self.find_calibration_versions(self.files.file_paths_1overf[filt][IMAGE_INDEX])

            if Path(self.files.file_paths_1overf_jhat[filt][IMAGE_INDEX]).exists() == False:
                self.assign_WCS_from_gaia_dr3(IMAGE_INDEX=IMAGE_INDEX, filter=filt)

            self.starbug2_aperture_photometry(  IMAGE_INDEX             = IMAGE_INDEX,
                                                filter                  = filt,
                                                remove_background       = True,
                                                toggle_sb_output_txt    = True,
                                                overwrite_always_run    = overwrite_always_run)

            self.aperture_data[filt]  = fits.open(self.files.file_paths_1overf_ap[filt][IMAGE_INDEX])[1].data

            self.wcs_align[filt], self.sky[filt]    = self._collect_wcs_sky_coords_of_SB2_detections(IMAGE_INDEX=IMAGE_INDEX,
                                            filter=filt,
                                            detection_sigma = detection_sigma
                                            )

            #self.plot_cal_steps_per_filter(IMAGE_INDEX=IMAGE_INDEX, filter=filt, save = save)



        _, _ , valid1, valid2   = self.cross_match_detections(self.sky[filt1], self.sky[filt2], det_tol=1e-5)

        filters = [filt1, filt2]
        validations = [valid1, valid2]
        self.magnitudes = {}
        for filt, valid in zip(filters, validations):
  
            self.magnitudes[filt] = self.aperture_data[filt][filt][valid]
            
        return

    def _collect_wcs_sky_coords_of_SB2_detections(self, IMAGE_INDEX=0, filter='F187N', detection_sigma=2.0):
        x_positions_sources, y_positions_sources, detection_mask    = self._collect_detections_mag_limited(filename_ap = self.files.file_paths_1overf_ap[filter][IMAGE_INDEX], filter=filter, detection_sigma= detection_sigma)
        align_fits = fits.open(self.files.file_paths_1overf_jhat[filter][IMAGE_INDEX])
        wcs_align = wcs.WCS(align_fits['SCI', 1], align_fits)
        sky = wcs_align.pixel_to_world(x_positions_sources[detection_mask], y_positions_sources[detection_mask])
        return wcs_align, sky


    def cross_match_detections(self, sky1, sky2, det_tol=1e-5):
        A = np.array([sky1.ra, sky1.dec]).T
        B = np.array([sky2.ra, sky2.dec]).T
   
        tree    = KDTree(B)
        matches = tree.query_ball_point(A, r=det_tol)

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


    def _overlap_indices(self, filters, det_tol):
            ref = filters[0]
            ref_idx = np.arange(len(self.sky[ref]))
            matched_mask = np.ones(len(ref_idx), dtype=bool)
    
            pair_matches = {}  # other_filter -> (a_idx into ref, b_idx into other)
            for other in filters[1:]:
                _, _, a_idx, b_idx = self.cross_match_detections(
                    self.sky[ref], self.sky[other], det_tol=det_tol
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
            print(prt + str(fits.open(filename)[0].header[f'{ext}']))
        return

    
    def _calibrate_one_over_f(self, IMAGE_INDEX, filt):

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


    def starbug2_aperture_photometry(self, IMAGE_INDEX, filter, remove_background = True, toggle_sb_output_txt = True, overwrite_always_run=False):
        '''
        TO DO: 
            - [] Add different parameter files based on filter

        '''
        starbug_path        = "/net/vdesk/data2/slagter/environment_path/SB_venv/bin/starbug2" 
 
        if remove_background == True:
            mode = "-vBD"
        else:
            mode = "-vD"

        param_cmd   =  [starbug_path, "-p",  self._determine_filename_sb2_param(filter)]

        cmd         = [starbug_path, mode, self.files.file_paths_1overf[filter][IMAGE_INDEX]]

        if overwrite_always_run == True:
                self._run_starbug2(param_cmd, cmd, toggle_sb_output_txt)

        else:
            
            if Path(self.files.file_paths_1overf_ap[filter][IMAGE_INDEX]).exists() == True and Path(self.files.file_paths_1overf_bgd[filter][IMAGE_INDEX]).exists()  == True: 
                print("Skipping starbug2 run")

                #If we dont want background this check is enough
            elif Path(self.files.file_paths_1overf_ap[filter][IMAGE_INDEX]).exists() == True and remove_background==False:
                print("Skipping starbug2 run")

            else:
                self._run_starbug2(param_cmd, cmd, toggle_sb_output_txt)

    def assign_WCS_from_gaia_dr3(self, IMAGE_INDEX=0, filter='F187N'):

        out_put_dir = '../../../../../../../..' + str(Path(self.files.file_paths[filter][IMAGE_INDEX]).parent.absolute().resolve())

        wcs_align.run_all(self.files.file_paths_1overf[IMAGE_INDEX],
                telescope='jwst',
                outsubdir=out_put_dir ,
                imagetype='cal',
                overwrite=True,
                d2d_max=.5,
                showplots=0,
                refcatname='Gaia',
                histocut_order='dxdy',
                    sharpness_lim=(0.3,0.9),
                    roundness1_lim=(-0.7, 0.7),
                    SNR_min= 3,
                    dmag_max=1.0,
                    objmag_lim =(14,24))

        return 
    
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    # (4) Helper functions for fits files
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    def _read_fits_collumn(self, filename,  target_data, col_num = 1, ):
        output = []
        for target_data_i in target_data:
            output.append(fits.open(filename)[col_num].data[f'{target_data_i}'])
        return output

    def _read_fits_detections(self, filename_ap, filter):                                                                    
            x_positions_sources = fits.open(filename_ap)[1].data[f'xcentroid']
            y_positions_sources = fits.open(filename_ap)[1].data[f'ycentroid']
            magnitude_error_sources = fits.open(filename_ap)[1].data['e'+ filter]
            return x_positions_sources, y_positions_sources, magnitude_error_sources

    def _collect_detections_mag_limited(self, filename_ap, filter, detection_sigma = 2.0):
        x_positions_sources, y_positions_sources, magnitude_error_sources = self._read_fits_detections(filename_ap = filename_ap, filter= filter)
        detection_mask              = magnitude_error_sources <= detection_sigma
        return x_positions_sources, y_positions_sources, detection_mask


    def _collect_image(self, filename):
        return fits.open(filename)[1].data
    
    def _determine_filename_sb2_param(self, filter):
        param_path  = "/net/vdesk/data2/slagter/environment_path/SB_venv/lib/python3.11/site-packages/starbug2/"
        return param_path + str('starbug' + '_' + filter + ".param")

    
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    # (5) Plotting functions
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

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
            plt.savefig(f'Plots/bgd_cal_steps_{filter}.pdf', bbox_inches='tight')
        plt.show()


    def show_detections(self,
                        filters=None,
                        mode='panel',
                        IMAGE_INDEX=28,
                        det_tol=1e-5,
                        save=True,
                        save_dir='Plots',
                        figsize_per_panel=(7, 7)):
        """
        Visualize STARBUG2 detections for one or more filters.

        """
        for filt in filters:
            if filt not in self.sky or filt not in self.wcs_align:
                raise KeyError(
                    f"'{filt}' has no cached sky coords / WCS yet - "
                    f"run run_pipeline(IMAGE_INDEX=...) for it first."
                )
    
        def _background(filt):
            image = self._collect_image(self.files.file_paths_1overf[filt][IMAGE_INDEX]) - self._collect_image(self.files.file_paths_1overf_bgd[filt][IMAGE_INDEX])
            vmin, vmax = ZScaleInterval().get_limits(image)
            return image, vmin, vmax
        
        def _style_ax(ax):
            ax.coords['dec'].set_ticks(spacing=10 * u.arcsec)
            ax.coords['ra'].set_ticks(spacing=10 * u.arcsec)
            ax.set_xlabel('')
            ax.set_ylabel('')
    
        # MODE: single filter, all detections
        if mode == 'single':
            filt = filters[0]
            image, vmin, vmax = _background(filt)
    
            fig = plt.figure(figsize=figsize_per_panel)
            gs = gridspec.GridSpec(1, 1, figure=fig)
            ax = fig.add_subplot(gs[0, 0], projection=self.wcs_align[filt])
    
            ax.imshow(image, cmap='bone', vmin=vmin, vmax=vmax, origin='lower')
            ax.scatter(self.sky[filt].ra, self.sky[filt].dec, s=35,
                    facecolors='none', edgecolors='cyan', linewidths=1,
                    label=f'{filt} detections',
                    transform=ax.get_transform('world'))
    
            _style_ax(ax)
            ax.legend(loc='upper right')
            fig.suptitle(filt)
    
            if save:
                fig.savefig(f'{save_dir}/detections_{filt}.pdf', bbox_inches='tight')
            plt.show()
            return fig
        
    
        # MODE: single panel, overlapping detections only
        if mode == 'overlap_only':
            for ref in filters:
                overlap_idx = self._overlap_indices(filters, det_tol )

        
                image, vmin, vmax = _background(ref)
        
                fig = plt.figure(figsize=figsize_per_panel)
                gs = gridspec.GridSpec(1, 1, figure=fig)
                ax = fig.add_subplot(gs[0, 0], projection=self.wcs_align[ref])
        
                ax.imshow(image, cmap='bone', vmin=vmin, vmax=vmax, origin='lower')
                ax.scatter(self.sky[ref].ra[overlap_idx[ref]], self.sky[ref].dec[overlap_idx[ref]],
                        s=35, facecolors='none', edgecolors='red', linewidths=1.5,
                        label=f'overlaping det',
                        transform=ax.get_transform('world'))
        
                _style_ax(ax)
                ax.legend(loc='upper right')
                fig.suptitle(f'Overlapping detections in {ref}')
        
                if save:
                    fig.savefig(f'{save_dir}/detections_overlap_{ref}.pdf', bbox_inches='tight')
                plt.show()

            return fig
    
        raise ValueError(f"Unknown mode '{mode}'. Use 'single', 'panel', or 'overlap_only'.")


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
        try:
            return fits.getval(cal_path, 'FILTER', ext=0)
        except Exception as e:
            print(f"Warning: couldn't read FILTER from {cal_path.name}: {e}")
            return None

    def _change_extentions(self, input_cal_file, mode='1overf'):
        if mode == '1overf':
            name = input_cal_file.replace('_cal.fits','_cal_1overf_cal.fits')
        elif mode == '1overf_bg':
            name = input_cal_file.replace('_cal.fits','_cal_1overf-bgd.fits')
        elif mode == '1overf_ap':
            name = input_cal_file.replace('_cal.fits','_cal_1overf-ap.fits')
        elif mode == 'jhat':
            name = input_cal_file.replace('_cal.fits','_cal_1overf_jhat.fits')

        else:
            print(f'{mode} is not valid')
            return
        return name
        
    def find_calibrated_filenames(self):

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