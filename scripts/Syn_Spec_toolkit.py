from astropy.table import Table
from sedfitter.sed import SEDCube

import numpy as np
import matplotlib.pyplot as plt

from astropy import units as u
from synphot import units, SourceSpectrum
from synphot.models import BlackBodyNorm1D, GaussianFlux1D

import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
from astropy.config import generate_config

generate_config(pkgname="synphot")
import numpy as np
import synphot
import astropy.constants as const
from tqdm import tqdm
from astropy.constants import Constant

# Fetch one filter's transmission curve
from astroquery.svo_fps import SvoFps
import numpy as np
from synphot import SpectralElement
from synphot.models import Empirical1D

from pathlib import Path
from synphot import Observation
import re
import warnings

from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpecFromSubplotSpec

from matplotlib.transforms import blended_transform_factory
from matplotlib.patches import ConnectionPatch
import matplotlib.ticker as ticker

from Config import MPLSTYLE, BESSEL_MODELS

if MPLSTYLE.exists():
    plt.style.use(str(MPLSTYLE))

wave_pa_alpha = 1.875 * u.um
wave_br_alpha = 4.051 * u.um
lam_pa = wave_pa_alpha.to(u.AA).value
lam_br = wave_br_alpha.to(u.AA).value

all_filters = [
    "F090W",
    "F115W",
    "F150W",
    "F182M",
    "F187N",
    "F200W",
    "F212N",
    "F277W",
    "F335M",
    "F356W",
    "F430M",
    "F444W",
    "F480M",
]


def create_wavelength_space_um():
    # 100 Å to ~3.16e6 Å, returned as a Quantity in µm
    return (np.logspace(2, 6.5, int(1e5)) * u.AA).to(u.um)


def Compute_FWHM_from_Weq(Weq):
    """
    Compute the FWHM from equavalent width ussuming line is GAUSSIAN

    Parameters
    ----------
    Weq :
        Equvalent width
    Returns
    -------
    FWHM
        Gaussian FWHM

    """
    return (2 * np.sqrt(2 * np.log(2) / np.pi)) * Weq


def finetuned_wavelength_space_um(
    lams_0=(wave_pa_alpha, wave_br_alpha), half_width_frac=0.02, n=2001
):
    grids = [create_wavelength_space_um()]
    for lam_0 in lams_0:
        fine = lam_0 * (1 + np.linspace(-half_width_frac, half_width_frac, n))
        grids.append(fine.to(u.um))
    return np.sort(np.concatenate(grids))


def _Normalize_flux(flux_old, d_new=200000 * u.lyr, d_old=1 * u.Rsun):
    return flux_old * (d_old / d_new).decompose().value ** (2)


def _Absolute_vega_mag_corr():
    """
    Offset between Flux at a 1 Rsun surface compared to 10 pc
    """
    return (
        -2.5 * 2 * np.log10((const.R_sun / (10 * u.pc)).decompose().value)
    ) * units.VEGAMAG


def _Obtain_transmission_curve(filter="F115W"):
    tbl = SvoFps.get_transmission_data(f"JWST/NIRCam.{filter}")

    return SpectralElement(
        Empirical1D,
        points=tbl["Wavelength"].quantity.to(u.AA),
        lookup_table=tbl["Transmission"].data,
    )


class Line_Strength:
    def __init__(self, continuum_mode, **kwargs):
        allowed_continuum_mode = {
            "Robitaille": self._Obtain_continuum_Robitaille,
            "Bessel": self._Obtain_continuum_Bessel,
            "Bessel + BB disk": self._Obtain_continuum_Bessel_plus_BB,
        }

        if continuum_mode not in allowed_continuum_mode:
            raise ValueError(
                f"'{continuum_mode}' is not valid. "
                f"Choose from: {list(allowed_continuum_mode)}"
            )

        self.sp_continuum = allowed_continuum_mode[continuum_mode](**kwargs)
        self.sp_tot = self.sp_continuum
        return

    def _reset_spectrum(
        self,
    ):
        """
        Erases any added line, returning to the specified continuum model
        """
        self.sp_tot = self.sp_continuum
        return

    def Compute_model_abs_MAG(self, spectrum, filter="F115W"):
        """
        Parameters
        ----------
        sp_tot : total spectrum to observe with the specified
                 Required to be at the flux density at the surface of 1 Rsun!
        filter : JWST filter to observe

        Returns
        -------
        Absolute magnitude of observation
        """
        bp = _Obtain_transmission_curve(filter=filter)

        # Convolce spectrum with the transmission curve
        # Note that the quantum efficiency is not required here (observing the same source with
        # and without excess efmission has the same losses which in the ∆m drop out!
        #
        # If goals is also estimation of exposure time --> pandeia!!!
        obs = Observation(spectrum, bp)

        # Computing the absolute magnitude of the observed spectrum
        return (
            obs.effstim(flux_unit=units.VEGAMAG) + _Absolute_vega_mag_corr()
        ).value  # NOTE THE SQUARED MAG UNIT

    def Add_emission_line_to_spectrum(self, lam0, Weq):
        F_cont = self.sp_continuum(lam0, flux_unit="FLAM")  # erg/s/cm2/AA
        total_flux = (Weq * F_cont).to(u.erg / u.s / u.cm**2)  # erg/s/cm2
        fwhm = Compute_FWHM_from_Weq(Weq=Weq)

        self.sp_line = SourceSpectrum(
            GaussianFlux1D, total_flux=total_flux, mean=lam0, fwhm=fwhm
        )

        self.sp_tot += self.sp_line
        return

    def _flam(self, sp):
        # attach the unit so synphot doesn't assume Angstrom
        return sp(self.wave_um, flux_unit="flam").value  # [erg/s/cm2/Å]

    def _lam_flam(self, sp):
        wave_AA = (self.wave_um).to(u.AA).value
        return wave_AA * self._flam(sp)  # [erg/s/cm2]

    def Plot_spectrum_modeled_source(self, Weq1, Weq2):
        def add_zoom(
            fig, ax_main, ax_zoom, x, y_cont, ytot, xlim, color_spine, Filter_range=None
        ):
            # force consistent units
            x0, x1 = (xlim[0].to(u.um).value, xlim[1].to(u.um).value)

            # Plot the same function
            ax_zoom.plot(x, ytot, lw=1, ls="-", c="blue")
            ax_zoom.plot(x, y_cont, lw=0.5, ls="-", c="black")

            # Add filter range:
            ax_zoom.axvline(Filter_range[0].to(u.um).value, ls=":", c="gray")
            ax_zoom.axvline(
                Filter_range[1].to(u.um).value,
                ls=":",
                c="gray",
                label=r"$T_{\mathrm{filter}}$",
            )

            ax_zoom.set_xlim(x0, x1)
            ymin, ymax = (0.9 * np.min(ytot), 1.1 * np.max(ytot))
            ax_zoom.set_ylim(ymin, ymax)
            ax_zoom.set_xlabel(r"$\lambda\ [\mu\mathrm{m}]$")
            ax_zoom.set_yscale("log")
            ax_zoom.legend()

            for s in ax_zoom.spines.values():
                s.set_edgecolor(color_spine)
                s.set_linewidth(1.5)

            transform_main = blended_transform_factory(
                ax_main.transData, ax_main.transAxes
            )

            for xa, xb in [(x0, 0), (x1, 1)]:
                fig.add_artist(
                    ConnectionPatch(
                        xyA=(xa, 0),
                        coordsA=transform_main,  # Exact op x0/x1 data-waarde, maar op de fysieke x-as (0)
                        xyB=(xb, 1),
                        coordsB=ax_zoom.transAxes,  # Linker- (0) en rechterbovenhoek (1) van de subplot
                        color=color_spine,
                        lw=1,
                        ls="--",
                    )
                )

        fig = plt.figure(figsize=(12, 8))
        gs = gridspec.GridSpec(
            2, 2, figure=fig, height_ratios=[2, 1], hspace=0.3, wspace=0.2
        )

        outer = gridspec.GridSpec(1, 1, figure=fig)[
            0
        ]  # or whichever cell of your gridspec
        inner = GridSpecFromSubplotSpec(
            2, 2, subplot_spec=outer, height_ratios=[3, 1], hspace=0.3, wspace=0.5
        )
        ax0 = fig.add_subplot(inner[0, :])
        ax1 = fig.add_subplot(inner[1, 0])
        ax2 = fig.add_subplot(inner[1, 1])

        self.wave_um = finetuned_wavelength_space_um()
        total_lam_flam = self._lam_flam(self.sp_tot)
        normalisation = np.nanmax(total_lam_flam)

        ax0.plot(
            self.wave_um,
            total_lam_flam / normalisation,
            "--",
            c="black",
            lw=1,
            label="Continuum",
        )
        for lam, name, col in [
            (0.3645, "Balmer break", "gray"),
            (0.6563, r"H$\alpha$", "gray"),
            (1.875, r"Pa$\alpha$", "tab:orange"),
            (4.051, r"Br$\alpha$", "tab:red"),
        ]:
            ax0.axvline(lam, lw=1, ls=":", color=col)
            ax0.text(lam, 3e-4, name, color=col, rotation=90, ha="right", va="bottom")

        delta_pa_alpja = Weq1 * 5
        Filter_range_pa = _Obtain_transmission_curve(filter="F187N").waverange

        wave_line1 = np.linspace(
            wave_pa_alpha - delta_pa_alpja, wave_pa_alpha + delta_pa_alpja, 10000
        ).to(u.um)

        ycont_1 = (
            wave_line1.to(u.AA).value
            * self.sp_continuum(wave_line1, flux_unit="flam").value
            / normalisation
        )
        ytot_1 = (
            wave_line1.to(u.AA).value
            * self.sp_tot(wave_line1, flux_unit="flam").value
            / normalisation
        )

        add_zoom(
            fig,
            ax0,
            ax1,
            wave_line1.value,
            y_cont=ycont_1,
            ytot=ytot_1,
            xlim=(wave_pa_alpha - delta_pa_alpja, wave_pa_alpha + delta_pa_alpja),
            color_spine="tab:orange",
            Filter_range=Filter_range_pa,
        )

        delta_br_alpja = Weq2 * 5
        Filter_range_br = _Obtain_transmission_curve(filter="F405N").waverange
        wave_line2 = np.linspace(
            wave_br_alpha - delta_br_alpja, wave_br_alpha + delta_br_alpja, 10000
        ).to(u.um)

        ycont_2 = (
            wave_line1.to(u.AA).value
            * self.sp_continuum(wave_line2, flux_unit="flam").value
            / normalisation
        )
        ytot_2 = (
            wave_line1.to(u.AA).value
            * self.sp_tot(wave_line2, flux_unit="flam").value
            / normalisation
        )

        add_zoom(
            fig,
            ax0,
            ax2,
            wave_line2.value,
            y_cont=ycont_2,
            ytot=ytot_2,
            xlim=(wave_br_alpha - delta_br_alpja, wave_br_alpha + delta_br_alpja),
            color_spine="tab:red",
            Filter_range=Filter_range_br,
        )

        ax0.set_xlim(0.1, 60)
        ax0.set_ylim(1e-4, 1e3)
        ax0.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

        ax0.set_xscale("log")
        ax0.set_yscale("log")
        ax0.set_xlabel(r"$\lambda\ [\mu\mathrm{m}]$")
        ax0.set_ylabel(r"$\lambda F_\lambda\ [\mathrm{erg\,s^{-1}\,cm^{-2}}]$")
        ax0.set_ylabel(r"$\lambda F_\lambda$ / max($\lambda F_\lambda^{\rm total}$)")
        ax0.legend()
        plt.show()
        return

    def _Obtain_continuum_Robitaille(
        self,
        parameter_range={
            "star.radius": (0.1, 10),
            "star.temperature": (4900, 5100),
            "disk.mass": None,
            "disk.rmax": None,
            "disk.beta": None,
            "disk.p": None,
            "disk.h100": None,
            "scattering": None,
            "inclination": (0.0, 10.0),  # Edge-on is 90 degrees
        },
    ):

        flux_path = (
            "/net/vdesk/data2/slagter/disc_models_robitaille_2017/sp--s-i/flux.fits"
        )
        disk_param_path = "/net/vdesk/data2/slagter/disc_models_robitaille_2017/sp--s-i/parameters.fits"
        param_table = Table.read(disk_param_path)
        seds = SEDCube.read(flux_path)

        x = 200000 * u.lyr * (0.046 * u.arcsec).to(u.rad).value
        ap_idx = np.argmin((np.abs(x - seds.apertures)))

        names = seds.names
        for param in parameter_range.keys():
            if parameter_range[param] != None:
                mask_i = (param_table[param] >= parameter_range[param][0]) & (
                    param_table[param] < parameter_range[param][1]
                )
                names[~mask_i] = "nan"

        # Reading the SED cube completely and in Synphot format using emperical 1d
        systems = {}

        cleaned_names = [x for x in names if str(x) != "nan"]
        for name in cleaned_names:
            sed = seds.get_sed(name)
            wave = sed.wav
            flux = _Normalize_flux(
                sed.flux[ap_idx], d_old=1 * u.kpc, d_new=200000 * u.lyr
            )
            systems[name] = SourceSpectrum(Empirical1D, points=wave, lookup_table=flux)

        if len(cleaned_names) > 1:
            warnings.warn(
                f" \n Parameter ranges include more than 1 model, showing first idx!!!"
            )
            sp_continuum = systems[cleaned_names[0]]

        if len(cleaned_names) < 1:
            warnings.warn(f"Warning...........Parameter ranges include 0 model!!!")
            sp_continuum = np.nan

        return sp_continuum

    def _Obtain_continuum_Bessel(self, T_eff=5000, g_surface="g45"):
        paths_and_temps = []
        for p in Path(BESSEL_MODELS).iterdir():
            if p.suffix == ".fits":
                match = re.search(r"\d{5}", p.name)
                if match:
                    paths_and_temps.append((p, float(match.group())))

        # Sort the list by the path name (or by temperature)
        paths_and_temps.sort(key=lambda x: x[0])

        # Unpack into separate tuples
        paths, temps = zip(*paths_and_temps) if paths_and_temps else ((), ())
        paths, temps = paths, np.array(temps)

        T_idx = np.argmin(np.abs(T_eff - temps))
        model_path_to_use = paths[T_idx]
        sp_continuum = SourceSpectrum.from_file(
            str(model_path_to_use), wave_col="WAVELENGTH", flux_col=g_surface
        )

        return sp_continuum

    def _Obtain_continuum_Bessel_plus_BB(self, T_eff=5000, g_surface="g45"):
        sp_star = self._Obtain_continuum_Bessel(T_eff=T_eff, g_surface=g_surface)

        A_ref = (np.pi * const.R_sun**2).to(u.m**2)
        R_in, R_out = [5, 20] * const.R_sun  # [0.1, 0.2] * u.au
        A_disk = (np.pi * (R_out**2 - R_in**2)).to(u.m**2)
        disc_scale = A_disk / A_ref
        T_disc = 1000 * u.K

        # F ∝ d^-2 -> F_disc = F_bb (d_bb/Rsun)^2 where d_bessel = 1 kpc
        # blackbody is per definition at Rsun at 1 kpc
        scale = ((const.kpc / const.R_sun).value) ** (2)
        bb_disc = SourceSpectrum(BlackBodyNorm1D, temperature=T_disc) * scale
        sp_disk = bb_disc * disc_scale
        return sp_star + sp_disk

def Compute_delta_m_weq(lam0, filters, N_Weq=100, continuum_mode="Bessel", **kwargs):
    SPEC = Line_Strength(continuum_mode=continuum_mode, **kwargs)

    MAGS_per_filter  = np.zeros((len(filters), 2, N_Weq))

    Weqs = np.logspace(0, 3.8, N_Weq, base=10) * u.AA

    for W_idx, W_eq in enumerate(tqdm(Weqs, desc="Computing ∆m(W_eq)", position=0)):
        SPEC._reset_spectrum()

        SPEC.Add_emission_line_to_spectrum(
            lam0=lam0,
            Weq=W_eq,
        )

        for F_idx, FILT in enumerate(filters):
            MAGS_per_filter[F_idx, 0, W_idx]  = SPEC.Compute_model_abs_MAG(spectrum=SPEC.sp_continuum, filter=FILT)
            MAGS_per_filter[F_idx, 1, W_idx] = SPEC.Compute_model_abs_MAG(spectrum=SPEC.sp_tot, filter=FILT)


    return Weqs, MAGS_per_filter


def Plot_delta_m_weq(Weqs, mag_diff_dic):
    fig = plt.figure(figsize=(7, 5))
    gs = gridspec.GridSpec(1, 1, figure=fig, hspace=0.3, wspace=0.2)
    ax = fig.add_subplot(gs[0, 0])

    for mag_diff in mag_diff_dic.keys():
        mag_diff_data, color = mag_diff_dic[mag_diff]
        ax.scatter(
            Weqs, mag_diff_data, c=color, edgecolors="black", marker="o", label=mag_diff
        )

    ax.axhline(0.3, ls="--", lw=2, c="black")

    ax.set_ylabel(r"$\Delta m = m_{\mathrm{no \ line}} - m_{\mathrm{line}}$")
    ax.set_xlabel(r"$W_{eq}$ [Å]")

    ax.set_xscale("log")
    ax.set_yscale("linear")

    ax.set_xlim((Weqs.value).min(), (Weqs.value).max())
    ax.set_ylim(-0.05, 1)

    ax.legend(loc="upper left")

    plt.show()
    return
