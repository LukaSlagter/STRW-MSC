import argparse
import os
from pathlib import Path
import photometric_pipeline as PPL
import phot_toolkit as PTK
from Config import (
    LEVEL3_DIR,
)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run StarBugII aperture photometry on a JWST NIRCam mosaic."
    )
    parser.add_argument(
        "--filter",
        required=True,
        help="JWST filter, e.g. F115W F187N, F335M, F277W, F200W, F444W, F182M, F430M, F090W, F356W, F150W, F480M, F212N",
    )
    parser.add_argument("--WHT_split", required=True, help="WHT weight to split filter on!")
    parser.add_argument("--ncpu", required=True, help="Number of cpu cores")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    filter_ = args.filter
    ncpu = args.ncpu
    mode = args.mode
    wht_threshold = args.WHT_split

    PHOTOMETRIC_pipeline = PPL.Pipeline_level_2_data(
        working_directory="/net/vdesk/data2/slagter/JWST_NIRCAM_#1227_level_2/mastDownload/JWST"
    )
    PTK.Split_Mosaic_on_WHT(mosaic_path = LEVEL3_DIR + f'/mosaic_{filter_}_final_i2d.fits',
                                wht_threshold = wht_threshold)

    for mode in ['short','long']:
        mosaic_path = f"/net/vdesk/data2/slagter/JWST_NIRCAM_#1227_level_3/mosaic_{filter_}_{mode}_exposure.fits"
        print(f"Will run Starbug2 on {mosaic_path}")
        if Path(mosaic_path).exists():
            PHOTOMETRIC_pipeline.starbug2_aperture_photometry_on_mosaic(
                mosaic_path=mosaic_path,
                filter=filter_,
                mode=mode,
                toggle_sb_output_txt=True,
                 N_CPUS=ncpu,
            )
        else:
            print(f"Path doesn't exist: {mosaic_path}")

    PHOTOMETRIC_pipeline.run_psf_on_individual_dithers(filt=filter_, N_CPUS=ncpu)
    return





