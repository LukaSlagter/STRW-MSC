from astropy.table import Table
from sedfitter.sed import SEDCube

import numpy as np
import matplotlib.pyplot as plt

from astropy import units as u
from synphot import units, SourceSpectrum
from synphot.models import BlackBodyNorm1D, GaussianFlux1D

import matplotlib.pyplot as plt,  matplotlib.gridspec as gridspec
from astropy.config import generate_config
generate_config(pkgname='synphot')
import numpy as np
import synphot
import astropy.constants as const

from astropy.constants import Constant
# Fetch one filter's transmission curve
from astroquery.svo_fps import SvoFps
import numpy as np
from synphot import SpectralElement
from synphot.models import Empirical1D

from pathlib import Path
from synphot import Observation
import re

filters = SvoFps.get_filter_list('JWST', instrument='NIRCam')

def thermal_Doppler_broadening(lam0, T =1000*u.K):
    m_hydrogen = const.m_e + const.m_p
    return (lam0 * np.sqrt(const.k_B * T /(m_hydrogen * const.c**2))).to(lam0.unit)


wave_pa_alpha = 1.875 * u.um
wave_br_alpha = 4.051 * u.um
lam_pa = wave_pa_alpha.to(u.AA).value
lam_br = wave_br_alpha.to(u.AA).value
sig_pa = thermal_Doppler_broadening(wave_pa_alpha.to(u.AA), 1000*u.K).value
sig_br = thermal_Doppler_broadening(wave_br_alpha.to(u.AA), 1000*u.K).value

wave = np.sort(np.concatenate([
    np.logspace(2, 6.5, int(1e5)),
    lam_pa + np.linspace(-20, 20, 2001)*sig_pa,
    lam_br + np.linspace(-20, 20, 2001)*sig_br,
])) * u.AA
wave_um = wave.to(u.um).value

from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.patches import ConnectionPatch
def add_zoom(fig, ax_main, ax_zoom, x, y, xlim, color):
    # Mask around the x and y reach of the function we zoom into
    x0, x1 = xlim
    x0, x1 = x0.value, x1.value
    m    = (x >= x0) & (x <= x1)
    ymin = y[m].min() / 2
    ymax = y[m].max() * 2
    ax_zoom.set_xlim(x0, x1)
    ax_zoom.set_ylim(ymin, ymax)

    #Plot the same function
    ax_zoom.plot(x, y, lw=1)
    ax_zoom.set_yscale('log')

    #give the spines of the plot a color
    for s in ax_zoom.spines.values():
        s.set_edgecolor(color); s.set_linewidth(1.5)
    # ax_main.add_patch(Rectangle((x0, ymin), x1 - x0, ymax - ymin,
    #                             fill=False, ec=color, lw=1.5))

    for xa, xb in [(x0, 0), (x1, 1)]:
        # Add the connecting arms
        fig.add_artist(ConnectionPatch(
            xyA=(xa, ymin), coordsA=ax_main.transData,
            xyB=(xb, 1),    coordsB=ax_zoom.transAxes,
            color=color, lw=1, ls="--"))


def plot_spectrum_modeled_source(sp_tot, sp_star, sp_disc, disc_scale, T_eff, T_disc):
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[2, 1], hspace=0.3, wspace=0.2)

    outer = gridspec.GridSpec(1, 1, figure=fig)[0]           # or whichever cell of your gridspec
    inner = GridSpecFromSubplotSpec(2, 2, subplot_spec=outer,
                                    height_ratios=[2, 1], hspace=0.3, wspace=0.2)
    ax0 = fig.add_subplot(inner[0, :])
    ax1 = fig.add_subplot(inner[1, 0])
    ax2 = fig.add_subplot(inner[1, 1])

    normalisation = np.max(lam_flam(sp_star + sp_disc))
    ax0.plot(wave_um, lam_flam(sp_tot)/normalisation,  '--', c='black', lw=1, label='Total')
    ax0.plot(wave_um, lam_flam(sp_star)/normalisation, lw=2, c='#33BBEE', label=f'Star (T = {T_eff:.0f} K)')
    ax0.plot(wave_um, lam_flam(sp_disc)/normalisation, lw=2, c='#EE3377', label=f'Disc (T = {T_disc:.0f}, scale×{disc_scale:.0f})')

    for lam, name in [ (0.6563, r'H$\alpha$'), (1.875, r'Pa$\alpha$'), (4.051, r'Br$\alpha$')]: #(0.3645, 'Balmer break'),
        ax0.axvline(lam, lw=1, ls=':', color='red')
        ax0.text(lam, 3e-4, name, color='red', rotation=90, ha='right', va='bottom')


    delta_pa_alpja = 100*thermal_Doppler_broadening(wave_pa_alpha, T = T_disc)
    wave_line1 = np.linspace(wave_pa_alpha - delta_pa_alpja,
                            wave_pa_alpha + delta_pa_alpja, 10000).to(u.um)
    
    y1 = wave_line1.to(u.AA).value * sp_tot(wave_line1, flux_unit='flam').value

    add_zoom(fig, ax0, ax1, wave_line1.value, y1/normalisation,
            (wave_pa_alpha - delta_pa_alpja, wave_pa_alpha + delta_pa_alpja), "tab:red")


    delta_br_alpja = 100*thermal_Doppler_broadening(wave_br_alpha, T = T_disc)

    wave_line2 = np.linspace(wave_br_alpha - delta_br_alpja,
                            wave_br_alpha + delta_br_alpja, 10000).to(u.um)
    
    y2 = wave_line2.to(u.AA).value * sp_tot(wave_line2, flux_unit='flam').value

    add_zoom(fig, ax0, ax2, wave_line2.value, y2/normalisation,
            (wave_br_alpha - delta_br_alpja, wave_br_alpha + delta_br_alpja), "darkred")

    ax0.set_xlim(0.05, 300)
    ax0.set_ylim(1e-4, 1e3)
    ax0.set_xscale('log'); ax0.set_yscale('log')
    ax0.set_xlabel(r'$\lambda\ [\mu\mathrm{m}]$')
    ax0.set_ylabel(r'$\lambda F_\lambda\ [\mathrm{erg\,s^{-1}\,cm^{-2}}]$')
    ax0.set_ylabel(r'$\lambda F_\lambda$ / max($\lambda F_\lambda^{\rm total}$)')
    ax0.legend()
    #plt.savefig('difficulties.pdf')
    plt.show()
    return

def flam(sp):
    return sp(wave, flux_unit='flam').value         # [erg/s/cm2/Å]

def bolometric(sp):
    return np.trapezoid(flam(sp), np.float128(wave.value))           # [erg/s/cm2]

def lam_flam(sp):
    return wave.value * flam(sp)                    # [erg/s/cm2]


def equivalent_width(d, sigma):
    '''plot_spectrum_modeled_source
    d     = fractional depth/heigth
    sigam = thermal broadening
    '''
    return np.sqrt(np.pi) * d * sigma

def rectangular_width_filter(filter):
    tbl     = SvoFps.get_transmission_data(f'JWST/NIRCam.{filter}')
    RW      = np.trapezoid(tbl['Transmission'], tbl['Wavelength']) / np.max(tbl['Transmission'])
    return RW* u.AA

def excess_magnitude(W, RW):
    return -2.5 * np.log10(1-W/RW)

def equivalent_width_from_magnitude(MAG_difference, RW):
    return RW * (10**(-0.4*MAG_difference)- 1)

def relative_gauss_amp(W, sigma):
    return W / (sigma* np.sqrt(2*np.pi))

def Obtain_line_Weq(lam0, W_eq, continuum):
    F_cont     = continuum(lam0, flux_unit='FLAM')           # erg/s/cm2/AA
    total_flux = (W_eq * F_cont).to(u.erg/u.s/u.cm**2)       # PHOTLAM * AA
    fwhm       = ( 2*np.sqrt(2* np.log(2)/ np.pi) )* W_eq
    
    return SourceSpectrum(GaussianFlux1D,
                          total_flux=total_flux,
                          mean      =lam0,
                          fwhm      =fwhm)

def Obtain_line_mag(lam0, MAG_difference, continuum,filter= 'F187N', T_disc =  1000*u.K):
    """
    OLD, idk if corrrect
    """
    RW               = rectangular_width_filter(filter)
    W                = equivalent_width_from_magnitude(MAG_difference, RW).to(u.AA)

    sigma            = thermal_Doppler_broadening(lam0  = lam0,
                                                   T    = T_disc).to(u.AA) 
    v_fwhm     = 100 *u.km / u.s
    sigma      = sigma = (lam0 * v_fwhm / const.c / 2.355).to(u.AA)

    total_flux = W * continuum(lam0)                 # PHOTLAM * AA
    fwhm       = 2*np.sqrt(2*np.log(2)) * sigma
    F_cont     = continuum(lam0, flux_unit='FLAM')           # erg/s/cm2/AA
    total_flux = (W * F_cont).to(u.erg/u.s/u.cm**2)

    return SourceSpectrum(GaussianFlux1D,
                          total_flux=total_flux,
                          mean=lam0,
                          fwhm=fwhm)

def _absolute_vega_mag_corr():
    '''
    Offset between Flux at a 1 Rsun surface compared to 10 pc
    '''
    return (-2.5*2*np.log10((const.R_sun/(10*u.pc)).decompose().value)) * units.VEGAMAG


def Compute_model_abs_MAG(sp, filter = 'F115W'):
    tbl = SvoFps.get_transmission_data(f'JWST/NIRCam.{filter}')
    bp = SpectralElement(
        Empirical1D,
        points=tbl['Wavelength'].quantity.to(u.AA),
        lookup_table=tbl['Transmission'].data,
    )
    obs= Observation(sp, bp)
    
    ABSOLUTE_MAGNITUDE = obs.effstim(flux_unit=units.VEGAMAG) + _absolute_vega_mag_corr()

    return ABSOLUTE_MAGNITUDE

def Find_available_models(directory='bessell_for_luka'):
    paths_and_temps = []
    for p in Path(directory).iterdir():
        if p.suffix == '.fits':
            match = re.search(r"\d{5}", p.name)
            if match:
                paths_and_temps.append((p, float(match.group())))
                
    # Sort the list by the path name (or by temperature)
    paths_and_temps.sort(key=lambda x: x[0])
    
    # Unpack into separate tuples
    paths, temps = zip(*paths_and_temps) if paths_and_temps else ((), ())
    
    return paths, np.array(temps)

def normalize_flux(flux_old, d_new = 200000 * u.lyr, d_old= 1*u.Rsun):
    return flux_old * (d_old/d_new).decompose().value**(2)






