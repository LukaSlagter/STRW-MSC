import os
from astroquery.mast import Observations
import sys

telescope = "JWST"
instrument_name = "NIRCAM*"
program_id = "1227"

folder = "/net/vdesk/data2/slagter/"
sys.path.append(folder)
dir_name = f"{telescope}_NIRCAM_#{program_id}_level_2"
os.makedirs(folder + dir_name, exist_ok=True)


print(f"Looking for program {program_id}...")

obs_table = Observations.query_criteria(
    obs_collection=telescope, instrument_name=instrument_name, proposal_id=program_id
)

print(f"Found {len(obs_table)} observations for #{program_id}.")

print("Generating product list...")
All_filters = [
    "F356W",
    "F150W",
    "F212N",
    "F187N",
    "F335M",
    "F444W",
    "F444W;F405N",
    "F430M",
    "F182M",
    "F444W;F470N",
    "F115W",
    "F480M",
    "F090W",
    "F277W",
    "F200W",
]
to_be_done_filters = [
    "F356W",
    "F150W",
    "F212N",
    "F335M",
    "F444W",
    "F444W;F405N",
    "F430M",
    "F182M",
    "F444W;F470N",
    "F480M",
    "F090W",
    "F277W",
    "F200W",
]


products = Observations.get_product_list(obs_table)


print("Filter for Level 2 images (_cal.fits)...")
filtered_products = Observations.filter_products(
    products,
    productSubGroupDescription="CAL",  # CAL --> Level 2 data
    extension="fits",
    filters=to_be_done_filters,
)

print(f"Found! {len(filtered_products)} Level 2 NIRCam files for #{program_id}.")

# Toon de eerste 10 resultaten in je terminal om te controleren
filtered_products[:10].pprint(max_lines=15)

# 4. DOWNLOADEN
total_bytes = sum(filtered_products["size"])
total_gb = total_bytes / (1024**3)
print(f"Carefull... total size is {total_gb:.1f} GB of data!")

Observations.download_products(filtered_products, download_dir=folder + dir_name)
