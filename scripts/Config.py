from pathlib import Path
import os

SCRATCH = Path("/home/lslagter/scratch/lslagter/MSC_project/STRW-MSC")
NET = Path("/net/vdesk/data2/slagter")

DATA_ROOT = SCRATCH if SCRATCH.exists() else NET
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHOT_DIR = PROJECT_ROOT / "Photometry"
CAT_DIR = PHOT_DIR / "catalogue_jeroen"
SB_PARAM_DIR = PHOT_DIR / "parameter_files_starbug"
BESSEL_MODELS = PROJECT_ROOT / "Continuum_emission_models/Synphot/Bessell_models"

PLOTS_DIR = PROJECT_ROOT / "Plots"

CATALOGUE_JEROEN = CAT_DIR / "NGC346_NIRCam_and_MIRI_Public_JWST_Catalogue.fits"
POSITIONS_F115W = CAT_DIR / "positions_F115W.fits"

GAIA_CATALOGUE = PROJECT_ROOT / "ngc346_gaia_dr3_10arcmin_clean.txt"
MPLSTYLE = PROJECT_ROOT.parent / "stylish.mplstyle"

LEVEL2_DIR = DATA_ROOT / "JWST_NIRCAM_#1227_level_2"
LEVEL3_DIR = DATA_ROOT / "JWST_NIRCAM_#1227_level_3"
STARBUG2_BIN = Path(
    os.environ.get("STARBUG2_BIN", DATA_ROOT / "environment_path/SB_venv/bin/starbug2")
)
