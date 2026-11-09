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
from scipy.interpolate import make_interp_spline
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

        # Sort data based on the C1-C3 color (Required for running median)
        self.sorting_sequence = np.argsort(
            (self.photometry_data[C1_name] - self.photometry_data[C3_name])
        )
        self.photometry_data = self.photometry_data[self.sorting_sequence]

        # Read the colors, sorted based on C1-C3
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

    def Set_template_Guido_2010(self, min_mag_diff=0.1, nsigma=4):
        # Fitted slopes to Bessel models!
        slope       =  0.9563743540982091
        intercept   = -0.02981862498051782
        def _fitted_func(xn):
            return slope * np.asarray(xn, float) + intercept
        
        self.template_C1_C2 = _fitted_func(self.C1_C3_color)
        self.Select_CC_outliers(min_mag_diff=min_mag_diff, nsigma=nsigma)
        return
    
    
    def _extrapolate_linear_tail(self, x_min, x_width=0.5):
        
        x = np.asarray(self.C1_C3_color, float)       
        y = np.asarray(self.running_median_C1_C2, float)    

        sel = np.isfinite(x) & np.isfinite(y) & (x > x_min - x_width) & (x <= x_min) & ~self.excess_mask

        slope, intercept = np.polyfit(x[sel], y[sel], deg=1)

        def _extrapolate_func(xn):
            return slope * np.asarray(xn, float) + intercept
        
        return _extrapolate_func

    def Set_template_Jeroen_2024(self, C_extrapolate =1.7, C_width=1,min_mag_diff=0.1, nsigma=4):
        self.Compute_rolling_median(window_size=100)       
        self.template_C1_C2 =  self.running_median_C1_C2
        self.Select_CC_outliers(min_mag_diff=min_mag_diff, nsigma=nsigma)

        f_extrapolate = self._extrapolate_linear_tail(x_min=C_extrapolate, x_width=C_width)

        mask = self.C1_C3_color > C_extrapolate

        template = np.array(self.running_median_C1_C2, dtype=float)   
        template[mask] = f_extrapolate(self.C1_C3_color[mask])

        self.template_C1_C2 = template

        self.Select_CC_outliers(min_mag_diff=min_mag_diff, nsigma=nsigma)
        return

    def Compute_rolling_median(self, window_size=1):
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
        self.running_median_C1_C2 = (
            pd.Series(self.C1_C2_color)
            .rolling(window_size, center=True, min_periods=1)
            .median()
            .to_numpy()
        )
        return

    def Select_CC_outliers(self, min_mag_diff, nsigma=4):
        """
        Excess if the C1-C2 colour exceeds the template by
        (a) > n sigma its own uncertainty
        AND
        (b) > min_mag_diff mag.
        """
        diff = self.C1_C2_color - self.template_C1_C2
        self.excess_mask = (diff > nsigma * self.C1_C2_eMag) & (diff > min_mag_diff)
        print(
            f"Found {np.sum(self.excess_mask)} out of {len(self.excess_mask)} excess sources"
        )
        return

    def EW_from_template(self, RW =0.024*u.um):
        
        self.EW = (RW * (1 - 10**(-0.4* (self.C1_C2_color - self.template_C1_C2)))).to(u.AA)
        self.EW_outliers = (self.EW[self.excess_mask]).to(u.AA)


        return 

