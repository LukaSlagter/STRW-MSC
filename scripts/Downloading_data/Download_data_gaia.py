# https://gaia.ari.uni-heidelberg.de/tap.html
# SELECT source_id, ra, dec, parallax, pmra, pmdec, phot_g_mean_mag, phot_g_mean_flux_over_error, phot_bp_mean_mag, phot_bp_mean_flux_over_error, phot_rp_mean_mag, phot_rp_mean_flux_over_error
# FROM gaiadr3.gaia_source
# WHERE 1=CONTAINS( POINT('ICRS', ra, dec), CIRCLE('ICRS', 14.770821063331482, -72.17833229841659, 0.17) )


import os
from astroquery.mast import Observations
import sys


from astroquery.gaia import Gaia
import astropy.units as u
from astropy.coordinates import SkyCoord
import numpy as np
import pandas as pd

telescope = "JWST"
instrument_name = "NIRCAM*"
program_id = "1227"

folder = "../../../../../../net/vdesk/data2/slagter/"
sys.path.append(folder)
dir_name = f"{telescope}_NIRCAM_#{program_id}_level_2"
os.makedirs(folder + dir_name, exist_ok=True)


GAIA_USER = "lslagter"
GAIA_PASS = "Xr7qVFLZBxPmWy@"

Gaia.login(user=GAIA_USER, password=GAIA_PASS)

# jobs = Gaia.list_async_jobs()
# print(f"Found {len(jobs)} jobs")

# job_ids = [j.jobid for j in jobs]
# print(job_ids)
# Gaia.remove_jobs(job_ids)


# Gaia.ROW_LIMIT = -1  # Find all sources --> NO CAP ON DOWNLOADED FILE SIZE
# Gaia.TIMEOUT = 120  # stop after 120 seconds of search??? ah connection time out

# # Central coordinates for NGC 346
# radius_arcmin = 10 #arcmin

# print(f"Targeting ngc 346...")
# ra_deg, dec_deg, radius_deg = (
#     14.770821063331482,
#     -72.17833229841659,
#     radius_arcmin / 60,
# )

# query = f"""
# SELECT source_id, ra, dec, parallax, pmra, pmdec,
#        phot_g_mean_mag, phot_g_mean_flux_over_error,
#         phot_bp_mean_mag, phot_bp_mean_flux_over_error,
#         phot_rp_mean_mag, phot_rp_mean_flux_over_error

# FROM gaiadr3.gaia_source
# WHERE 1=CONTAINS(
#     POINT('ICRS', ra, dec),
#     CIRCLE('ICRS', {ra_deg}, {dec_deg}, {radius_deg})
# )
# """

# job = Gaia.launch_job_async(query, verbose=True)
# results = job.get_results()


# df = results.to_pandas()  # putting masked values --> NaN
# df.to_csv(
#     folder + dir_name + f"/ngc346_gaia_dr3_{radius_arcmin}arcmin_clean.txt",
#     sep=" ",
#     index=False,
#     na_rep="NaN",
# )
# Gaia.remove_jobs([job.jobid])
