import argparse
import os
from pathlib import Path

os.chdir("/home/slagter/Astronomy_master/STRW-MSC/SF_in_NGC")

import photometric_pipeline as PPL


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


def main() -> None:
    args = parse_args()
    filter_ = args.filter
    ncpu = args.ncpu

    PHOTOMETRIC_pipeline = PPL.Pipeline_level_2_data(
        working_directory="/net/vdesk/data2/slagter/JWST_NIRCAM_#1227_level_2/mastDownload/JWST"
    )

    PHOTOMETRIC_pipeline.run_psf_on_individual_dithers(filt=filter_, N_CPUS=ncpu)


if __name__ == "__main__":
    main()
