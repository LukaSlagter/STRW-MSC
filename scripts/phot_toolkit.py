# ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= =======
# Author    : Luka Slagter
# Goal      : Data pipeline for level 2 JWST observations of NGC 346 (proposal #1227)
# ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= =======
import os, sys, subprocess
import tqdm
import numpy as np

import astropy.units as u, astropy.constants as const
from astropy.table import Table
from astropy.io import fits
from astropy.coordinates import SkyCoord
from Config import (
    LEVEL2_DIR,
    LEVEL3_DIR,
)

# ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= =======
# Reading Jeroens catalogue + masking region
# ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= ======= =======
class Read_Catalogue:
    def __init__(
        self,
        catalogue_path="/home/slagter/Astronomy_master/STRW-MSC/SF_in_NGC/Photometry/NGC346_NIRCam_and_MIRI_Public_JWST_Catalogue.fits",
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
            (self.cat_data["RA"] >= RA_range[0])
            & (self.cat_data["RA"] <= RA_range[1])
            & (self.cat_data["DEC"] <= DEC_range[1])
            & (self.cat_data["DEC"] >= DEC_range[0])
        )
        return

    def mask_sources_on_FILTERer(self, FILTERer, detection_limit=2.0):
        self.cat_FILTERer_mask = self.cat_data["e" + FILTERer] < detection_limit
        return

    def make_source_list(self, FILTER_mask_on=None, outfile="Photometry/test_pos.fits"):
        tab = Table.read(self.catalogue_path, hdu=1)

        if FILTER_mask_on == None:
            new = tab[["Catalogue_Number", "RA", "DEC"]]
        else:
            if type(FILTER_mask_on) == tuple:
                self.mask_sources_on_FILTERer(
                    FILTER_mask_on[0], detection_limit=FILTER_mask_on[1]
                )
                new = tab[["Catalogue_Number", "RA", "DEC"]][self.cat_FILTERer_mask]  #
            else:
                return print("HELP")

        new["RA"].unit = "deg"
        new["DEC"].unit = "deg"
        # Build the BinTable HDU with a minimal primary header
        hdu = fits.table_to_hdu(new)
        hdu.header["OBJECT"] = "NGC 346"
        hdu.header["ORIGIN"] = "JWST NIRCam+MIRI observations"
        hdu.header.add_comment(
            "RA/DEC (ICRS) extracted from cleaned JWST photometric catalogue"
        )

        fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(outfile, overwrite=True)
        return outfile


def load_starbug_params(file_path):
    """
    READS StarBugII .param file
    """
    params = {}

    with open(file_path, "r") as file:
        for line in file:
            clean_line = line.strip()

            if not clean_line or clean_line.startswith("//"):
                continue

            # Starbug2 : 'key = value // comment'
            if "=" in clean_line:
                key_part, rest = clean_line.split("=", 1)
                key = key_part.strip()

                # Remoce inline comments
                value_part = rest.split("//")[0].strip()

                # Datatype convertion (float, int of string)
                try:
                    if "." in value_part:
                        value = float(value_part)
                    else:
                        value = int(value_part)
                except ValueError:
                    value = value_part

                params[key] = value

    return params


def find_center(ny, nx, wcs):
    return wcs.pixel_to_world((nx - 1) / 2, (ny - 1) / 2)


def make_target_wcs(center, dra, ddec, rot, width, height, pix_scale):
    from astropy.wcs import WCS

    new_center = center.spherical_offsets_by(dra, ddec)
    n_x = int((width / pix_scale).decompose())
    n_y = int((height / pix_scale).decompose())
    th = rot.to_value(u.rad)

    target = WCS(naxis=2)
    target.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    target.wcs.crval = [new_center.ra.deg, new_center.dec.deg]
    target.wcs.crpix = [n_x / 2 + 0.5, n_y / 2 + 0.5]
    target.wcs.cdelt = [-pix_scale.to_value(u.deg), pix_scale.to_value(u.deg)]
    target.wcs.pc = [[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]
    return target, (n_y, n_x)


def in_wcs_mask(ra, dec, wcs, shape):
    """
    Returns a boolean mask, True where the source lies inside the footprint.
    """
    ra = u.Quantity(ra, u.deg)
    dec = u.Quantity(dec, u.deg)
    n_y, n_x = shape
    x, y = wcs.world_to_pixel(SkyCoord(ra, dec, frame="icrs"))

    mask = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= -0.5)
        & (x <= n_x - 0.5)
        & (y >= -0.5)
        & (y <= n_y - 0.5)
    )
    return mask



def Split_Mosaic_on_WHT(FILTER='F115W', wht_threshold=200.0
):
    mosaic_path=str(LEVEL3_DIR) + f"/mosaic_{FILTER}_final_i2d.fits"
    hdu = fits.open(mosaic_path)

    # Build masks & Apply
    wht_data = hdu["WHT"].data
    long_exposure_mask = wht_data > wht_threshold
    short_exposure_mask = ~long_exposure_mask
    long_exposure_data = []
    short_exposure_data = []

    for fit_frame in [
        "SCI",
        "ERR",
        "WHT",
        "VAR_POISSON",
        "VAR_RNOISE",
        "VAR_FLAT",
    ]:  # con
        data = hdu[fit_frame].data
        header = hdu[fit_frame].header
        long_exposure_data.append(
            fits.ImageHDU(
                data=np.where(long_exposure_mask, data, np.nan).astype(data.dtype),
                header=header,
                name=fit_frame,
            )
        )

        short_exposure_data.append(
            fits.ImageHDU(
                data=np.where(short_exposure_mask, data, np.nan).astype(data.dtype),
                header=header,
                name=fit_frame,
            )
        )
    print('happy')

    hdu_out = fits.HDUList([fits.PrimaryHDU(header=hdu[0].header)] + long_exposure_data)
    hdu_out.writeto(
        str(LEVEL3_DIR) + f"/mosaic_{FILTER}_long_exposure.fits", overwrite=True
    )

    hdu_out = fits.HDUList(
        [fits.PrimaryHDU(header=hdu[0].header)] + short_exposure_data
    )
    hdu_out.writeto(
        str(LEVEL3_DIR) + f"/mosaic_{FILTER}_short_exposure.fits", overwrite=True
    )

    hdu.close()
    print("----Completed-Splitting-Mosaic----")
    return

import re
def merge_catalogues(long_file, short_file, out_file, match_arcsec=0.06, max_err=2.0):
    def load(f):

        found = [m.group(1) for c in t.colnames if (m := re.fullmatch(r"e(F\d{3}[WMN])", c))]
        if filt is None:
            if len(found) == 1:
                filt = found[0]
        
        t = Table.read(f, hdu=1)
        mask = np.isfinite(err) & (t[f"e{filt}"] < max_err)
        return t[mask]

    deep, shallow = load(long_file), load(short_file)

    c_deep = SkyCoord(deep["RA"], deep["DEC"], unit="deg")
    c_shal = SkyCoord(shallow["RA"], shallow["DEC"], unit="deg")

    _, sep, _ = c_shal.match_to_catalog_sky(c_deep)
    keep = sep > match_arcsec * u.arcsec

    merged = np.vstack([deep, shallow[keep]], join_type="outer")
    merged.write(out_file, overwrite=True)

    print(f"deep: {len(deep)}, shallow: {len(shallow)}, "
          f"duplicates dropped: {(~keep).sum()}, merged: {len(merged)}")
    return merged