import sys
from pathlib import Path

scripts_path = str(Path.cwd().parents[0]) + "/STRW-MSC/scripts/"
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)
print(scripts_path)

from Config import (
    LEVEL2_DIR,
)

import photometric_pipeline as PPL

working_directory = f"{LEVEL2_DIR}/mastDownload/JWST"
PHOTOMETRIC_pipeline = PPL.Pipeline_level_2_data(working_directory=working_directory)


def main() -> None:
    for filt in ["F115W", "F187N", "F405N","F090W","F150W","F182M","F212N","F277W","F335M","F356W","F430M","F444W","F480M"]:
                        
        PHOTOMETRIC_pipeline.Run_AP_on_all_dither(
            filt=filt,
            overwrite_dic={
                "1/f": False,
                "Jhat calibration": False,
                "Association file": True,
                "Image3Pipeline": True,
            },
        )


if __name__ == "__main__":
    main()
