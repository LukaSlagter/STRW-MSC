import argparse
import os
from pathlib import Path

os.chdir("/home/slagter/Astronomy_master/STRW-MSC/SF_in_NGC")

import photometric_pipeline as PPL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run StarBugII aperture photometry on a JWST NIRCam mosaic."
    )
    parser.add_argument("--filter", required=True, help="JWST filter, e.g. F115W F187N, F335M, F277W, F200W, F444W, F182M, F430M, F090W, F356W, F150W, F480M, F212N")
    parser.add_argument("--mode", default="long", help="Mosaic mode, e.g. long or short (default: long). Mosaic should have been split before hand!!!")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    filter_ = args.filter 
    mode = args.mode

    mosaic_path = (
        f"/net/vdesk/data2/slagter/JWST_NIRCAM_#1227_level_3/mosaic_{filter_}_{mode}_exposure.fits"
    )
    print(f'Will run Starbug2 on {mosaic_path}')

    PHOTOMETRIC_pipeline = PPL.Pipeline_level_2_data(
        working_directory="/net/vdesk/data2/slagter/JWST_NIRCAM_#1227_level_2/mastDownload/JWST"
    )

    if Path(mosaic_path).exists():
        PHOTOMETRIC_pipeline.starbug2_aperture_photometry_on_mosaic(
            mosaic_path=mosaic_path,
            filter=filter_,
            mode=mode,
            toggle_sb_output_txt=True,
        )
    else:
        print(f"Path doesn't exist: {mosaic_path}")


if __name__ == "__main__":
    main()