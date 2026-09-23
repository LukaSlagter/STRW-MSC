import os
from astroquery.mast import Observations
import sys


from astroquery.gaia import Gaia
import astropy.units as u
from astropy.coordinates import SkyCoord
import numpy as np
import pandas as pd

telescope           = "JWST"
instrument_name     = "NIRCAM*"
program_id          = "1227"

folder = "../../../../../../net/vdesk/data2/slagter/"
sys.path.append(folder)
dir_name     = f'{telescope}_NIRCAM_#{program_id}_level_2'
os.makedirs(folder+dir_name, exist_ok=True) 


#print(f"Looking for program {program_id}...")

# obs_table = Observations.query_criteria(
#     obs_collection=telescope,
#     instrument_name=instrument_name,
#     proposal_id=program_id
# )

# print(f"Found {len(obs_table)} observations for #{program_id}.")

# print("Generating product list...")
# products = Observations.get_product_list(obs_table)


# print("Filter for Level 2 images (_cal.fits)...")
# filter = "F115W"
# filtered_products = Observations.filter_products(
#     products,
#     productSubGroupDescription="CAL",  # CAL --> Level 2 data
#     extension="fits",
#     filters=filter
# )

# print(f"Found! {len(filtered_products)} Level 2 NIRCam files for #{program_id}.")

# # Toon de eerste 10 resultaten in je terminal om te controleren
# filtered_products[:10].pprint(max_lines=15)

# # 4. DOWNLOADEN
# total_bytes = sum(filtered_products['size'])
# total_gb = total_bytes / (1024**3)
# print(f"Carefull... total size is {total_gb:.1f} GB of data!")

# Observations.download_products(filtered_products, download_dir=folder+dir_name)





Gaia.ROW_LIMIT  = -1     # Find all sources --> NO CAP ON DOWNLOADED FILE SIZE
Gaia.TIMEOUT    = 120      # stop after 120 seconds of search???

# Central coordinates for NGC 346
print(f"Targeting ngc 346...")
ra_deg, dec_deg, radius_deg = 14.770821063331482, -72.17833229841659, 5/60  # 5 arcmin in deg

query = f"""
SELECT source_id, ra, dec, parallax, pmra, pmdec,
       phot_g_mean_mag, phot_g_mean_flux_over_error,
        phot_bp_mean_mag, phot_bp_mean_flux_over_error,
        phot_rp_mean_mag, phot_rp_mean_flux_over_error

FROM gaiadr3.gaia_source
WHERE 1=CONTAINS(
    POINT('ICRS', ra, dec),
    CIRCLE('ICRS', {ra_deg}, {dec_deg}, {radius_deg})
)
"""

job = Gaia.launch_job_async(query, verbose=True)
results = job.get_results()
 

prefac      = 2.5/np.log(10) # 2.5/ln(10)
sigmaG_0    = 0.0027553202
sigmaGBP_0  = 0.0027901700
sigmaGRP_0  = 0.0037793818
results["phot_g_mean_mag_error"]  = prefac / results["phot_g_mean_flux_over_error"]
results["phot_bp_mean_mag_error"] = prefac / results["phot_bp_mean_flux_over_error"]
results["phot_rp_mean_mag_error"] = prefac / results["phot_rp_mean_flux_over_error"]


# after computing the *_mag_error columns
df = results.to_pandas()          # masked values -> NaN
df.to_csv(folder + dir_name + '/ngc346_gaia_dr3_5arcmin_clean.txt',
          sep=' ', index=False, na_rep='NaN')   # header=True by default
Gaia.remove_jobs([job.jobid])