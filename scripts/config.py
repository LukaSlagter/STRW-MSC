from pathlib import Path
import os

PROJECT_ROOT = Path(os.environ.get("SF_IN_NGC_ROOT", Path(__file__).resolve().parents[1]))

# Large data lives elsewhere (cluster disk), so it gets its own override
DATA_ROOT = Path(os.environ.get("SF_IN_NGC_DATA", "/net/vdesk/data2/slagter"))

PHOT_DIR      = PROJECT_ROOT / "Photometry"
CAT_DIR       = PHOT_DIR / "catalogue_jeroen"
SB_PARAM_DIR  = PHOT_DIR / "parameter_files_starbug"
PLOTS_DIR     = PROJECT_ROOT / "Plots"

CATALOGUE_JEROEN = CAT_DIR / "NGC346_NIRCam_and_MIRI_Public_JWST_Catalogue.fits"
POSITIONS_F115W  = CAT_DIR / "positions_F115W.fits"

GAIA_CATALOGUE   = PROJECT_ROOT / "ngc346_gaia_dr3_10arcmin_clean.txt"   # wherever you keep it
MPLSTYLE         = PROJECT_ROOT.parent / "stylish.mplstyle"

LEVEL3_DIR   = DATA_ROOT / "JWST_NIRCAM_#1227_level_3"
STARBUG2_BIN = Path(os.environ.get("STARBUG2_BIN",
                    DATA_ROOT / "environment_path/SB_venv/bin/starbug2"))
