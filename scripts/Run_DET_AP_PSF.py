import argparse
import os
from pathlib import Path
import photometric_pipeline as PPL
import phot_toolkit as PTK

from Config import (


    LEVEL2_DIR,
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

print('-------------------------TEST1 -------------------------')

def main() -> None:
    args = parse_args()
    filter_ = str(args.filter)
    ncpu = args.ncpu
    wht_threshold = args.WHT_split


    
    working_directory = f"{LEVEL2_DIR}/mastDownload/JWST"
    PHOTOMETRIC_pipeline = PPL.Pipeline_level_2_data(working_directory=working_directory)
    # PTK.Split_Mosaic_on_WHT(mosaic_path = str(LEVEL3_DIR) + f'/mosaic_{filter_}_final_i2d.fits', filt=filter_,
    #                             wht_threshold = int(wht_threshold))
    # print('-------------------------Finsisehd WHT SPLIT-------------------------')

    # for mode in ['short','long']:
    #     mosaic_path = str(LEVEL3_DIR) + f"/mosaic_{filter_}_{mode}_exposure.fits"
    #     print(f"Will run Starbug2 on {mosaic_path}")
    #     if Path(mosaic_path).exists():
    #         PHOTOMETRIC_pipeline.starbug2_aperture_photometry_on_mosaic(
    #             mosaic_path=mosaic_path,
    #             filter=filter_,
    #             mode=mode,
    #             toggle_sb_output_txt=True,
    #             N_CPUS=ncpu,
    #         )
    #     else:
    #         print(f"Path doesn't exist: {mosaic_path}")

    for filter in PHOTOMETRIC_pipeline.files.file_paths_1overf_jhat.keys():
        print(f"{filter}: {len(PHOTOMETRIC_pipeline.files.file_paths_1overf_jhat[filter])}")
    PHOTOMETRIC_pipeline.run_psf_on_individual_dithers(filt=filter_, N_CPUS=ncpu)
    return



if __name__ == "__main__":
    main()



