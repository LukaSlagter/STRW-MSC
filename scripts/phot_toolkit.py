# ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= =======
# Author    : Luka Slagter
# Goal      : Data pipeline for level 2 JWST observations of NGC 346 (proposal #1227)
# ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= =======
import os, sys, subprocess
import tqdm

from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
from matplotlib.collections import EllipseCollection
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from matplotlib.patches import Circle
from matplotlib.colors import LogNorm
plt.style.use("/home/slagter/Astronomy_master/STRW-MSC/stylish.mplstyle")


import astropy.units as u, astropy.constants as const
from astropy.table import Table
from astropy.visualization import ZScaleInterval
from astropy.io import fits
from astropy import wcs
from astropy.wcs.utils import proj_plane_pixel_scales, skycoord_to_pixel
from astropy.stats import SigmaClip
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.coordinates import SkyCoord; from astropy.nddata import extract_array; from astropy.visualization import simple_norm, LinearStretch
import jhat
os.chdir("/home/slagter/Astronomy_master/STRW-MSC/SF_in_NGC")
os.environ["CRDS_PATH"] = os.path.expanduser("~/crds_cache")
os.environ["CRDS_CONTEXT"] = "jwst_1584.pmap"
os.environ["CRDS_SERVER_URL"] = "https://jwst-crds.stsci.edu"

import jwst; from jwst.pipeline import Image3Pipeline; from jwst.associations import asn_from_list; from jwst.associations.lib.rules_level3_base import DMS_Level3_Base

# is now in the .background / .segmentation thingy...
from photutils.background import Background2D, MedianBackground; from photutils.segmentation import detect_sources, make_2dgaussian_kernel; from jwst.datamodels import dqflags




# ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= =======
# Paramters
# ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= =======

wcs_align = jhat.st_wcs_align()
verbose = 2
wcs_align.verbose = verbose
new_line = "\n"

# ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= =======
# Reading Jeroens catalogue + masking region
# ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= =======

class Read_Catalogue:
    def __init__(
        self,
        catalogue_path="Photometry/NGC346_NIRCam_and_MIRI_Public_JWST_Catalogue.fits",
    ):
        self.catalogue_path = catalogue_path
        self.catalogue = fits.open(catalogue_path)
        self.cat_data = self.catalogue[1].data
        return

    def show_headers(self):
        """
        show headers of the catalogue
        """
        with fits.open(self.catalogue_path) as hdulist:
            for i, hdu in enumerate(hdulist):
                print(f"--- HEADER EXTENSIE {i} ({hdu.name}) ---")
                print(repr(hdu.header))
                print("\n" + "=" * 40 + "\n")
        return

    def mask_region(self, RA_range, DEC_range):
        """
        Zoom in to specified region (cutout in RA and DEC)
        """
        self.cat_mask = ( 
            (self.cat_data["RA"] >= RA_range[0]) & (self.cat_data["RA"] <= RA_range[1]) & \
            (self.cat_data["DEC"] <= DEC_range[1]) & (self.cat_data["DEC"] >= DEC_range[0]))
        return

    def mask_sources_on_filter(self, filter, detection_limit = 2.0):
        self.cat_filter_mask = self.cat_data['e'+filter] < detection_limit
        return

    def make_source_list(self, filt_mask_on=None , outfile = 'Photometry/test_pos.fits'):
        tab = Table.read(self.catalogue_path, hdu=1)

        if filt_mask_on == None:
            new         = tab[["Catalogue_Number", "RA", "DEC"]]
        else:
            if type(filt_mask_on) == tuple:
                self.mask_sources_on_filter(filt_mask_on[0],  detection_limit = filt_mask_on[1])
                new         =  tab[self.cat_filter_mask]#tab[["Catalogue_Number", "RA", "DEC"]][self.cat_filter_mask] # 
            else:
                return print('HELP')
        
        new["RA"].unit = "deg"
        new["DEC"].unit = "deg"
        # Build the BinTable HDU with a minimal primary header
        hdu = fits.table_to_hdu(new)
        hdu.header["OBJECT"]  = "NGC 346"
        hdu.header["ORIGIN"]  = "JWST NIRCam+MIRI observations"
        hdu.header.add_comment("RA/DEC (ICRS) extracted from cleaned JWST photometric catalogue")
        
        fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(outfile, overwrite=True)
        return outfile


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

def find_center(ny, nx, wcs ):
    return wcs.pixel_to_world((nx - 1) / 2, (ny - 1) / 2)

def make_target_wcs(center, dra, ddec, rot, width, height, pix_scale):
    from astropy.wcs import WCS
    new_center = center.spherical_offsets_by(dra, ddec)
    n_x = int((width  / pix_scale).decompose())
    n_y = int((height / pix_scale).decompose())
    th  = rot.to_value(u.rad)

    target = WCS(naxis=2)
    target.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    target.wcs.crval = [new_center.ra.deg, new_center.dec.deg]
    target.wcs.crpix = [n_x / 2 + 0.5, n_y / 2 + 0.5]
    target.wcs.cdelt = [-pix_scale.to_value(u.deg), pix_scale.to_value(u.deg)]
    target.wcs.pc    = [[np.cos(th), -np.sin(th)],
                        [np.sin(th),  np.cos(th)]]
    return target, (n_y, n_x)


def in_wcs_mask(ra, dec, wcs, shape):
    """
    Returns a boolean mask, True where the source lies inside the footprint.
    """
    ra  = u.Quantity(ra, u.deg)
    dec = u.Quantity(dec, u.deg)
    n_y, n_x = shape
    x, y = wcs.world_to_pixel(SkyCoord(ra, dec, frame='icrs'))

    mask = (
        np.isfinite(x) & np.isfinite(y) &
        (x >= -0.5) & (x <= n_x - 0.5) &
        (y >= -0.5) & (y <= n_y - 0.5)
    )
    return mask





