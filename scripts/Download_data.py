import os
from astroquery.mast import Observations
import sys
import timeit


telescope           = "JWST"
instrument_name     = "NIRCAM*"
program_id          = "1227"

print(f"Looking for program {program_id}...")

obs_table = Observations.query_criteria(
    obs_collection=telescope,
    instrument_name=instrument_name,
    proposal_id=program_id
)

print(f"Found {len(obs_table)} observations for #{program_id}.")

print("Generating product list...")
products = Observations.get_product_list(obs_table)


print("Filter for Level 2 images (_cal.fits)...")
filter = "F115W"
filtered_products = Observations.filter_products(
    products,
    productSubGroupDescription="CAL",  # CAL --> Level 2 data
    extension="fits",
    filters=filter
)

print(f"Found! {len(filtered_products)} Level 2 NIRCam files for #{program_id}.")

# Toon de eerste 10 resultaten in je terminal om te controleren
filtered_products[:10].pprint(max_lines=15)

# 4. DOWNLOADEN
total_bytes = sum(filtered_products['size'])
total_gb = total_bytes / (1024**3)
print(f"Carefull... total size is {total_gb:.1f} GB of data!")

folder = "../../../../../../net/vdesk/data2/slagter/"
sys.path.append(folder)
dir_name     = f'{telescope}_{instrument_name}_#{program_id}_level_2/{filter}/'

os.makedirs(folder+dir_name, exist_ok=True) 
Observations.download_products(filtered_products, download_dir=folder+dir_name)
