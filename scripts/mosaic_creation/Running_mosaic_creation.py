import sys
from pathlib import Path

scripts_path = str(Path.cwd().parents[1]) + "/scripts/"
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)


from Config import (
    LEVEL2_DIR,
)

import photometric_pipeline as PPL

working_directory = f"{LEVEL2_DIR}/mastDownload/JWST"
PHOTOMETRIC_pipeline = PPL.Pipeline_level_2_data(working_directory=working_directory)


def main() -> None:

    PHOTOMETRIC_pipeline.calibrate_all_individual_dithers(
        filt="F115W",
        overwrite_dic={
            "1/f": True,
            "Jhat calibration": True,
            "Association file": True,
            "Image3Pipeline": True,
        },
    )


if __name__ == "__main__":
    main()
