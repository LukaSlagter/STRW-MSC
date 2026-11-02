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
    parser.add_argument("--ncpu", required=True, help="Number of cpu cores")
    return parser.parse_args()

working_directory = f"{LEVEL2_DIR}/mastDownload/JWST"
PHOTOMETRIC_pipeline = PPL.Pipeline_level_2_data(working_directory=working_directory)

all_filters = {
    # "F090W": '?',
    "F115W": 200,
    # "F150W": '?',
    # "F182M": '?',
    "F187N": 100,
    "F200W": 200,
    # "F212N": '?',
    "F277W": '?',
    "F335M": '?',
    # "F356W": '?',
    # 'F405N': '?',
    # "F430M": '?',
    "F444W": 100,
    # "F480M": '?',
}

def main() -> None:
    args = parse_args()
    FILTER = str(args.filter)
    ncpu = args.ncpu

    def _complete_run(FILTER):

        PHOTOMETRIC_pipeline.Calibrate_all_dithers(FILTER=FILTER,
                overwrite_dic={
                    "1/f": False,
                    "Jhat calibration": False,
                },)

        PHOTOMETRIC_pipeline.Create_Mosaic(FILTER=FILTER,
                            overwrite_dic={
                                "Association file": False,
                                "Image3Pipeline file": False,
                            },
                            output_dir=LEVEL3_DIR)
        
        Mosiac_ap_path = PHOTOMETRIC_pipeline.Compute_Aperture_Photometry_Mosaic( FILTER= FILTER, wht_threshold = all_filters[FILTER], ncpu = ncpu)
        print(f'Finished aperture photometry for {FILTER}')
        return Mosiac_ap_path

        
    if FILTER == 'ALL':
        for FILTER_ in all_filters.keys():
            _complete_run(FILTER= FILTER_)
    
    else:
        _complete_run(FILTER= FILTER)


    #PHOTOMETRIC_pipeline.run_psf_on_individual_dithers(filt=FILTER, N_CPUS=ncpu)
    return



if __name__ == "__main__":
    main()



