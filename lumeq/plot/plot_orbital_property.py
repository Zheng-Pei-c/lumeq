from lumeq import os, sys, np
from lumeq.property.orbital_property import cal_property_from_scratch

from lumeq.plot import plt, mcolors, ticker
from lumeq.plot import brokenaxes

def plot_orbital_property(fig_name, orbital_property):
    fig = plt.figure(figsize=(12, 3), dpi=300, layout='constrained')

    keys = ['energy', 'velocity', 'velocity2', 'velocity3']

    nrow, ncol = 1, len(keys)-1
    gs = fig.add_gridspec(nrow, ncol)
    #subfig = fig.subfigures(nrow, ncol, wspace=3, hspace=1)

    energy = orbital_property[keys[0]]
    xlims = ((energy[0]-5, energy[1]+5), (energy[2]-2, energy[-1]+2))

    for i in range(nrow*ncol):
        values = orbital_property[keys[i+1]]
        ylims = [np.min(values[:2]), np.max(values[:2]), np.min(values[2:]), np.max(values[2:])]
        if ylims[0] < ylims[2]:
            ylims = [ylims[2], ylims[3], ylims[0], ylims[1]]
        margin = [(ylims[1]-ylims[0])/100, (ylims[3]-ylims[2])/100]
        ylims = ((ylims[0]-margin[0], ylims[1]+margin[0]), (ylims[2]-margin[1], ylims[3]+margin[1]))
        ratio = [[1,2],[1,2]]
        pad = [.1,.1]

        axs = brokenaxes(None, gs[i], xlims, ylims, ratio=ratio, pad=pad)

        for ax in axs:
            ax.plot(energy, values, marker='.', mfc='r', mec='r')
            if i==0:
                ax.ticklabel_format(axis='y', style='sci', scilimits=(5,1))

            ax.set_xlabel(keys[0]+' ('+orbital_property['units'][keys[0]]+')')
        axs[0].set_ylabel(keys[i+1]+' ('+orbital_property['units'][keys[i+1]]+')')
        axs[2].set_ylabel(keys[i+1]+' ('+orbital_property['units'][keys[i+1]]+')')

#        subfig[i].supxlabel(keys[0]+' ('+orbital_property['units'][keys[0]]+')', y=0)
#        subfig[i].supylabel(keys[i+1]+' ('+orbital_property['units'][keys[i+1]]+')')

    plt.tight_layout()
    plt.savefig(fig_name)


if __name__ == '__main__':
    infile = sys.argv[1]
    orbital_property = cal_property_from_scratch(infile)

    fig_name = infile.split('.in')[0]
    plot_orbital_property(fig_name, orbital_property)
