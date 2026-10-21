import numpy as np

import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
from matplotlib.collections import EllipseCollection
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from matplotlib.patches import Circle
from matplotlib.colors import LogNorm
plt.style.use("/home/slagter/Astronomy_master/STRW-MSC/stylish.mplstyle")

def _make_mag_name(filter):
    return fr'$m_{{\mathrm{{{str(filter)}}}}}$'


def plot_colorspace_diagram(tables, mode='CC',filtx1= 'F200Wmag', filtx2='F444Wmag', filty1='F115Wmag', filty2='F182Mmag', xlim = (-10,10), ylim = (-10,10)):
    Zsun    = 0.0152
    fig = plt.figure()
    gs = gridspec.GridSpec(1,1, figure=fig)
    ax  = fig.add_subplot(gs[0,0])

    for table in tables:

        if mode == 'CC':
            plt.title(f'Color-Color diagram - CMD 3.9 models')
            ax.set_xlabel(f'{_make_mag_name(filtx1)} - {_make_mag_name(filtx2)}')
            ax.set_ylabel(f'{_make_mag_name(filty1)} - {_make_mag_name(filty2)}')
            Fx1_Fx2 = table[filtx1] - table[filtx2] 
            Fy1_Fy2 = table[filty1] - table[filty2] 
           
        elif mode == 'CM':
            plt.title(f'Color-Magnitude diagram - CMD 3.9 models')
            ax.set_xlabel(f'{_make_mag_name(filtx1)} - {_make_mag_name(filtx2)}')
            ax.set_ylabel(f'{_make_mag_name(filty1)}')
            Fx1_Fx2 = table[filtx1] - table[filtx2] 
            Fy1_Fy2 = table[filty1]

        # !!
        # Assuming we used constant metalicity and ages!!!!!
        # !!!

        logAge  = np.min(table['logAge'])
        Zini    = np.min(table['Zini'])

        Age_colors = {'6.0': '#33BBEE', '8.0' : "#EF8A0E", '10.0': '#EE3377'}
        Z_marker   = {str(Zsun): 'o', str(Zsun/5): 'x'}
        Z_label    = {str(Zsun): r'$Z_{\odot}$', str(Zsun/5): r'$Z_{\odot}/5$'}
        ax.scatter(Fx1_Fx2, Fy1_Fy2, marker=Z_marker[str(Zini)], label=fr'$\log_{{10}} [\mathrm{{t_{{age}}}}]$ = {logAge}, Z =  {Z_label[str(Zini)]}', color = Age_colors[str(logAge)])
       
   
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.legend()
        
    plt.show()
    return