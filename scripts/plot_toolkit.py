import numpy as np

import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
from matplotlib.collections import EllipseCollection
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from matplotlib.patches import Circle
from matplotlib.colors import LogNorm

plt.style.use("/home/slagter/Astronomy_master/STRW-MSC/stylish.mplstyle")

from astropy.io import fits
import astropy.units as u
from astropy.visualization import ZScaleInterval
from astropy.wcs import WCS
from reproject import reproject_interp
import matplotlib.patches as mpatches
from astropy.stats import SigmaClip
from astropy.convolution import Gaussian2DKernel, convolve

import phot_toolkit as PTK
import photometric_pipeline as PPL
from astropy.coordinates import SkyCoord
from matplotlib.colors import to_rgb
from astropy.visualization import (
    ZScaleInterval,
    AsinhStretch,
    ImageNormalize,
    make_lupton_rgb,
)
from astropy.wcs.utils import proj_plane_pixel_scales
from photutils.aperture import CircularAperture

def _make_mag_name(filter):
    return rf"$m_{{\mathrm{{{str(filter)}}}}}$"


def plot_colorspace_diagram(
    tables,
    mode="CC",
    filtx1="F200Wmag",
    filtx2="F444Wmag",
    filty1="F115Wmag",
    filty2="F182Mmag",
    xlim=(-10, 10),
    ylim=(-10, 10),
):
    Zsun = 0.0152
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 1, figure=fig)
    ax = fig.add_subplot(gs[0, 0])

    for table in tables:

        if mode == "CC":
            plt.title(f"Color-Color diagram - CMD 3.9 models")
            ax.set_xlabel(f"{_make_mag_name(filtx1)} - {_make_mag_name(filtx2)}")
            ax.set_ylabel(f"{_make_mag_name(filty1)} - {_make_mag_name(filty2)}")
            Fx1_Fx2 = table[filtx1] - table[filtx2]
            Fy1_Fy2 = table[filty1] - table[filty2]

        elif mode == "CM":
            plt.title(f"Color-Magnitude diagram - CMD 3.9 models")
            ax.set_xlabel(f"{_make_mag_name(filtx1)} - {_make_mag_name(filtx2)}")
            ax.set_ylabel(f"{_make_mag_name(filty1)}")
            Fx1_Fx2 = table[filtx1] - table[filtx2]
            Fy1_Fy2 = table[filty1]

        # !!
        # Assuming we used constant metalicity and ages!!!!!
        # !!!

        logAge = np.min(table["logAge"])
        Zini = np.min(table["Zini"])

        Age_colors = {"6.0": "#33BBEE", "8.0": "#EF8A0E", "10.0": "#EE3377"}
        Z_marker = {str(Zsun): "o", str(Zsun / 5): "x"}
        Z_label = {str(Zsun): r"$Z_{\odot}$", str(Zsun / 5): r"$Z_{\odot}/5$"}
        ax.scatter(
            Fx1_Fx2,
            Fy1_Fy2,
            marker=Z_marker[str(Zini)],
            label=rf"$\log_{{10}} [\mathrm{{t_{{age}}}}]$ = {logAge}, Z =  {Z_label[str(Zini)]}",
            color=Age_colors[str(logAge)],
        )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.legend()

    plt.show()
    return


# def _draw_markers(ax, x, y, r, color):
#     """Detections; points are dropped if not in frame"""

#     ax.add_collection(
#         EllipseCollection(
#             2 * r,
#             2 * r,
#             0,
#             units="xy",
#             offsets=np.c_[x, y],
#             offset_transform=ax.transData,  # transOffset= on matplotlib < 3.6
#             fc="none",
#             ec=color,
#             lw=1,
#         )
#     )

#     return

def plot_detection_in_frame(
    Image_path,
    source_coordinates,
    save_name="Plots/XXX.pdf",
    center = None,
    tag=None,
    dra=1 * u.arcmin,
    ddec=0 * u.arcmin,
    rot=0 * u.deg,
    width=2 * u.arcmin,
    height=2 * u.arcmin,
    pix_scale=0.5 * u.arcsec,
):

    with fits.open(Image_path, memmap=True) as hdu:
        data = hdu["SCI"].data
        wcs = WCS(hdu["SCI"].header, hdu)
        if center == None:
            center = PTK.find_center(*data.shape, wcs=wcs)
        target_wcs, shape_out = PTK.make_target_wcs(
            center, dra, ddec, rot, width, height, pix_scale
        )

        img, _ = reproject_interp((data, wcs), target_wcs, shape_out=shape_out)

    fig = plt.figure(figsize=(5, 5))
    gs = gridspec.GridSpec(1, 1, figure=fig)  # once, before the loop
    ax = fig.add_subplot(gs[0, 0], projection=target_wcs)

    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(img[np.isfinite(img)])

    cmap = "gray"
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")

    if source_coordinates != None:
        mask = PTK.in_wcs_mask(
            source_coordinates.ra, source_coordinates.dec, target_wcs, shape_out
        )
        x, y = target_wcs.world_to_pixel(source_coordinates[mask])
        n_y, n_x = shape_out


        pixel_scales = proj_plane_pixel_scales(target_wcs)
        pixel_scale_deg = pixel_scales * u.deg / u.pixel

        radius_angle = 0.1 * u.arcsec
        radius_pixel = (radius_angle.to(u.deg) / pixel_scale_deg[0]).value

        apertures = CircularAperture(np.array([x,y]).T, r=radius_pixel)
        apertures.plot(color='#ff004c', lw=1)

        if tag is not None:
            tag = np.asarray(tag, dtype=bool)
            tag_mask = mask & tag
            xt, yt = target_wcs.world_to_pixel(source_coordinates[tag_mask])
            aperturest = CircularAperture(np.array([xt,yt]).T, r=radius_pixel)
            aperturest.plot(color="#0ee7ff", lw=1)

    ax.set_facecolor(plt.colormaps[cmap](0))
    ax.coords["ra"].set_axislabel("RA")
    ax.coords["dec"].set_axislabel("Dec")
    ax.grid(color="w", alpha=0.3)
    if save_name:
        plt.savefig(save_name, bbox_inches="tight")
    return target_wcs, shape_out


def find_center(ny, nx, wcs ):
    return wcs.pixel_to_world((nx - 1) / 2, (ny - 1) / 2)

def make_target_wcs(center, dra, ddec, rot, width, height, pix_scale):
    

    new_center = center.spherical_offsets_by(dra, ddec)
    n_x = int((width  / pix_scale).decompose())
    n_y = int((height / pix_scale).decompose())
    th  = rot.to_value(u.rad)

    target = WCS(naxis=2)
    target.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    target.wcs.crval = [new_center.ra.deg, new_center.dec.deg]
    target.wcs.crpix = [n_x / 2 + 0.5, n_y / 2 + 0.5]
    target.wcs.cdelt = [-pix_scale.to_value(u.deg), pix_scale.to_value(u.deg)]
    target.wcs.pc    = [[np.cos(th), -np.sin(th)],
                        [np.sin(th),  np.cos(th)]]
    return target, (n_y, n_x)

def load_image_color(filt, target, shape_out):
    """
    Reproject single mosaic into grid
    """
    level3_data_path = '/net/vdesk/data2/slagter/JWST_NIRCAM_#1227_level_3/'
    with fits.open(level3_data_path + f'mosaic_{filt}_final_i2d.fits', memmap=True) as hdu:
        data = hdu['SCI'].data
        w    = WCS(hdu['SCI'].header, hdu)
        img, _ = reproject_interp((data, w), target, shape_out=shape_out)
    return img


def plot_mosaic(filter      = 'F115W',
                center      = SkyCoord(ra=14.95176547*u.deg, dec=-72.21625529*u.deg,frame='icrs'), #the one from F187N
                dra         = 1 * u.arcmin, 
                ddec        = 0 * u.arcmin,
                rot         = 0 * u.deg,                    
                width       = 15 * u.arcmin,                
                height      = 15 * u.arcmin,
                pix_scale   = 0.5 * u.arcsec,             
                ):

    # center = find_center(*Image.shape, wcs= wcs_mosaic)
    # print(center)
    target, shape_out = make_target_wcs(center, dra, ddec, rot, width, height, pix_scale)
    Image             = load_image_color(filter, target, shape_out)

    fig = plt.figure(figsize=(7,7))
    gs = gridspec.GridSpec(1, 1, figure=fig)  # once, before the loop
    ax = fig.add_subplot(gs[0, 0], projection=target)

    interval            = ZScaleInterval()
    vmin, vmax          = interval.get_limits(Image)

    ax.imshow(Image, cmap='bone', vmin=vmin, vmax=vmax, origin='lower')
    ax.set_facecolor(plt.colormaps['bone'](0))
    ax.coords['ra'].set_axislabel('RA')
    ax.coords['dec'].set_axislabel('Dec')
    ax.grid(color='w', alpha=0.3)
    plt.savefig(f"Plots/Mosaic_{filter}_final.pdf", bbox_inches='tight')
    return


def load_split_image_color(filt, target, shape_out, mode='long'):
    """
    Reproject single mosaic into grid
    """
    level3_data_path = '/net/vdesk/data2/slagter/JWST_NIRCAM_#1227_level_3/'
    with fits.open(level3_data_path + f'mosaic_{filt}_{mode}_exposure.fits', memmap=True) as hdu:
        data = hdu['SCI'].data
        w    = WCS(hdu['SCI'].header, hdu)
        img, _ = reproject_interp((data, w), target, shape_out=shape_out)
    return img

def plot_mosaic_split(filter      = 'F115W',
                mode        = 'long',
                center      = SkyCoord(ra=14.95176547*u.deg, dec=-72.21625529*u.deg,frame='icrs'), #the one from F187N
                dra         = 1 * u.arcmin, 
                ddec        = 0 * u.arcmin,
                rot         = 0 * u.deg,                    
                width       = 15 * u.arcmin,                
                height      = 15 * u.arcmin,
                pix_scale   = 0.5 * u.arcsec,             
                ):

    # center = find_center(*Image.shape, wcs= wcs_mosaic)
    # print(center)
    target, shape_out = make_target_wcs(center, dra, ddec, rot, width, height, pix_scale)
    Image             = load_split_image_color(filter, target, shape_out,mode = mode)

    fig = plt.figure(figsize=(7,7))
    gs = gridspec.GridSpec(1, 1, figure=fig)  # once, before the loop
    ax = fig.add_subplot(gs[0, 0], projection=target)

    interval            = ZScaleInterval()
    vmin, vmax          = interval.get_limits(Image)

    ax.imshow(Image, cmap='bone', vmin=vmin, vmax=vmax, origin='lower')
    ax.set_facecolor(plt.colormaps['bone'](0))
    ax.coords['ra'].set_axislabel('RA')
    ax.coords['dec'].set_axislabel('Dec')
    ax.grid(color='w', alpha=0.3)
    plt.savefig(f"Plots/Mosaic_{filter}_final.pdf", bbox_inches='tight')
    return

def plot_mosaic_rgb(plot_param = {  "F444W": [(1.0, 0.0, 0.0), 1], # Red
                                    "F335M": [(1.0, 1.0, 0.0), 1], # Yellow
                                    "F277W": [(0.0, 1.0, 0.0), 0.2], # Green
                                    "F200W": [(0.0, 0.0, 1.0), 0.2], # Blue
                                    "F187N": [(0.0, 1.0, 1.0), 0.1], # Cyan
                                    "F115W": [(1.0, 0.0, 1.0), 1], # Magenta
                                },
                    center    = SkyCoord(ra=14.95176547*u.deg, dec=-72.21625529*u.deg, frame='icrs'),
                    dra       = 1 * u.arcmin,
                    ddec      = 0 * u.arcmin,
                    rot       = 0 * u.deg,
                    width     = 15 * u.arcmin,
                    height    = 15 * u.arcmin,
                    pix_scale = 0.5 * u.arcsec,
                    ):

    filters             = list(plot_param.keys())
    
    colors, weights     = [plot_param[filter][0] for filter in filters], [plot_param[filter][1] for filter in filters]
    target, shape_out = make_target_wcs(center, dra, ddec, rot, width, height, pix_scale)
    # Format into the targed grid
    images_per_filter = [load_image_color(f, target, shape_out) for f in filters]
    
    # Build three color channels
    R = sum(w * c[0] * im for w, c, im in zip(weights, colors, images_per_filter))
    G = sum(w * c[1] * im for w, c, im in zip(weights, colors, images_per_filter))
    B = sum(w * c[2] * im for w, c, im in zip(weights, colors, images_per_filter))

    vmin, vmax = np.nanpercentile(np.stack([R, G, B]), [10, 99.7])
    rgb = make_lupton_rgb(R, G, B,
                        minimum=vmin,
                        stretch=(vmax - vmin)*0.5,
                        Q=5)
    
    fig = plt.figure(figsize=(7, 7))
    ax  = fig.add_subplot(111, projection=target)
    ax.imshow(rgb, origin='lower')
    ax.coords['ra'].set_axislabel('RA')
    ax.coords['dec'].set_axislabel('Dec')
    ax.grid(color='w', alpha=0.3)
    fig.savefig(f"Plots/Mosaic_{'_'.join(filters)}_RGB.pdf", bbox_inches='tight')

    return 
