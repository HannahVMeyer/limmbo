
# coding: utf-8
######################
### import modules ###
######################

import h5py

# plotting
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import colors, cm

import scipy as sp
import numpy as np
import pylab as pl
import os


###########################
### functions: plotting ###
###########################


def plot_heatmaps(Cg_init, Cn_init, Cg_predicted, Cn_predicted, seed, outdir,
        nr_plots=5, cb_labsize=12, ax_labsize=12):
    limit_cg = np.fabs([Cg_init.min(), Cg_init.max(),
        np.array(Cg_predicted[:(nr_plots - 2)]).min(),
        np.array(Cg_predicted[:(nr_plots - 2)]).max()]).max()
    limit_cn = np.fabs([Cn_init.min(), Cn_init.max(),
        np.array(Cn_predicted[:(nr_plots - 2)]).min(),
        np.array(Cn_predicted[:(nr_plots - 2)]).max()]).max()

    cmap = cm.PRGn
    norm_cg = colors.Normalize(-limit_cg, limit_cg)
    norm_cn = colors.Normalize(-limit_cn, limit_cn)
    cb_labsize = cb_labsize
    ax_labsize = ax_labsize

    pl.figure(1, figsize=(20, 20))
    pl.subplot(nr_plots, 2, 1)
    pl.title('Cg_init')
    pl.imshow(Cg_init, vmin=-limit_cg, vmax=limit_cg, cmap=cmap, norm=norm_cg)
    plot_custom(cmap=cmap, norm=norm_cg,
                cb_labsize=cb_labsize, ax_labsize=ax_labsize)
    pl.subplot(nr_plots, 2, 2)
    pl.title('Cn_init')
    pl.imshow(Cn_init, vmin=-limit_cn, vmax=limit_cn, cmap=cmap, norm=norm_cn)
    plot_custom(cmap=cmap, norm=norm_cn,
                cb_labsize=cb_labsize, ax_labsize=ax_labsize)
    for i in range(0, nr_plots, 2):
        pl.subplot(nr_plots, 2, 3 + i)
        pl.title('Cg_predicted')
        pl.imshow(Cg_predicted[i], vmin=-limit_cg,
                  vmax=limit_cg, cmap=cmap, norm=norm_cg)
        plot_custom(cmap=cmap, norm=norm_cg,
                    cb_labsize=cb_labsize, ax_labsize=ax_labsize)
        pl.subplot(nr_plots, 2, 3 + i + 1)
        pl.title('Cn_predicted')
        pl.imshow(Cn_predicted[i], vmin=-limit_cn,
                  vmax=limit_cn, cmap=cmap, norm=norm_cn)
        plot_custom(cmap=cmap, norm=norm_cn,
                    cb_labsize=cb_labsize, ax_labsize=ax_labsize)

    pl.tight_layout()
    figname = 'mtSet_reconstruct_%s.pdf' % seed
    figname = os.path.join(outdir, figname)
    pl.savefig(figname)
    pl.show()
    pl.clf()


def plot_custom(cmap, norm, cb_labsize, ax_labsize):
    cb = pl.colorbar(cmap=cmap, norm=norm)
    cb.ax.tick_params(labelsize=cb_labsize)
    pl.tick_params(axis='both', labelsize=ax_labsize)
