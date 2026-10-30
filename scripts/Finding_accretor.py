import sys
from pathlib import Path

scripts_path = str(Path.cwd().parents[1]) + "/scripts/"
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

from Config import (
    CATALOGUE_JEROEN,
    POSITIONS_F115W,
    GAIA_CATALOGUE,
    MPLSTYLE,
    SB_PARAM_DIR,
    PHOT_DIR,
    PLOTS_DIR,
    LEVEL2_DIR,
    LEVEL3_DIR,
    STARBUG2_BIN,
)
import phot_toolkit as PTK
import photometric_pipeline as PPL
import plot_toolkit as PLOT

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
from matplotlib.colors import to_rgb
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import astropy.units as u

if MPLSTYLE.exists():
    plt.style.use(str(MPLSTYLE))

working_directory = f"{LEVEL2_DIR}/mastDownload/JWST"
PHOTOMETRIC_pipeline = PPL.Pipeline_level_2_data(working_directory=working_directory)



class Accretion_Finder:
    def __init__(self, photometry_data, max_mag_err, C1_name, C2_name, C3_name):
        names = (C1_name, C2_name, C3_name)

        # Mean error of the combined filters!
        sigma_i = np.array([photometry_data["e" + n] for n in names])
        sigma_3 = np.sqrt(np.mean(sigma_i**2, axis=0))

        # Apply maximum magnitude error cut
        self.photometry_data = photometry_data[sigma_3 < max_mag_err]

        # Sort data based on the C1-C2 color (Required for running median & makes sure we dont forget)
        self.sorting_sequence = np.argsort((self.photometry_data[C1_name]-self.photometry_data[C2_name]))
        self.photometry_data = self.photometry_data[self.sorting_sequence]

        self.C1, self.C2, self.C3 = (self.photometry_data[n] for n in names)
        self.eC1, self.eC2, self.eC3 = (self.photometry_data["e" + n] for n in names)        
        self.C1_C2_color = self.C1 - self.C2
        self.C1_C3_color = self.C1 - self.C3

        # hypot(x1, x2) -> sqrt(x1**2 + x2**2)
        # Assumes independent errors!!!
        # => breaks down when EW ~> 100 Å
        self.C1_C2_eMag = np.hypot(self.eC1, self.eC2)
        self.C1_C3_eMag = np.hypot(self.eC1, self.eC3)
        return

    def Select_CC_outliers(self, min_mag_diff, nsigma=4):
        n_sigma_below_data = self.C1_C2_color - (nsigma * self.C1_C2_eMag)
        self.excess_mask = n_sigma_below_data > (self.rolling_median_C1_C2 + min_mag_diff)
        print(f'Found {np.sum(self.excess_mask)} out of {len(self.excess_mask)} excess sources')
        return 


    def Compute_rolling_median(self, window_size = 10000):
        """
        example:
        s = pd.Series([0, 1, 2, 3, 4])
        s.rolling(3, center=True, min_periods=1).median()
        0    0.5
        1    1.0
        2    2.0
        3    3.0
        4    3.5
        dtype: float64
        
        """
        window_size = window_size
    
        # Compute the rolling median
        # Center = True ensures the median computes is for the index we are at
        # min_periods sets the first [0, 1, ..., window_size/2] values to nan
        self.rolling_median_C1_C2 = (
            pd.Series(self.C1_C2_color)
            .rolling(window_size, center=True, min_periods=1)
            .median()
            .to_numpy()
        )
        return
