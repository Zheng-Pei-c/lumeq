from lumeq import np
from lumeq.plot import plt, mcolors, ticker, LineCollection

from scipy.stats import gaussian_kde, norm

def get_kwargs(marker):
    return dict(marker=marker, markersize=10,
                linestyle='none', color='k', mec='k', mew=1, clip_on=False)


def brokenaxes(fig, gs, xlims, ylims, ratio=[1.,1.], pad=.1):
    func = fig.add_gridspec if gs==None else gs.subgridspec
    nr = len(ylims) if isinstance(ylims, list) else 1
    nc = len(xlims) if isinstance(xlims, list) else 1

    if nr == 1:
        gs = func(nr, nc, width_ratios=ratio, wspace=pad)
    elif nc == 1:
        gs = func(nr, nc, height_ratios=ratio, hspace=pad)
    else:
        gs = func(nr, nc, width_ratios=ratio[0], height_ratios=ratio[1],
                  wspace=pad[0], hspace=pad[1])

    axs = gs.subplots()

    if nr == 1: # split xaxis
        for i in range(nc):
            axs[i].set_xlim(xlims[i])
            axs[i].set_ylim(ylims)
            if i>0: axs[i].set_yticks([])
            #axs[i].tick_params(labelright=False)
        #axs[-1].yaxis.tick_right()
    elif nc == 1: # split yaxis
        for i in range(nr):
            axs[i].set_xlim(xlims)
            axs[i].set_ylim(ylims[i])
            if i<nr-1: axs[i].set_xticks([])
    else: # split both axes
        for i in range(nr):     # y
            for j in range(nc): # x
                axs[i,j].set_xlim(xlims[j])
                axs[i,j].set_ylim(ylims[i])
                if j>0: axs[i,j].set_yticks([])
                if i<nr-1: axs[i,j].set_xticks([])

    for ax in axs.flat:
        ss = ax.get_subplotspec()
        if nr > 1:
            ax.spines['top'].set_visible(ss.is_first_row())
            ax.spines['bottom'].set_visible(ss.is_last_row())
        if nc > 1:
            ax.spines['left'].set_visible(ss.is_first_col())
            ax.spines['right'].set_visible(ss.is_last_col())


    d = .5 # how big to make the diagonal lines in axes coordinates
    if nr == 1:
        marker=[(-d, -1), (d, 1)]
        kwargs = get_kwargs(marker)
        axs[0].plot([1, 1], [1, 0], transform=axs[0].transAxes, **kwargs)
        axs[1].plot([0, 0], [1, 0], transform=axs[1].transAxes, **kwargs)
    elif nc == 1:
        marker=[(-1, -d), (1, d)]
        kwargs = get_kwargs(marker)
        axs[0].plot([0, 1], [0, 0], transform=axs[0].transAxes, **kwargs)
        axs[1].plot([0, 1], [1, 1], transform=axs[1].transAxes, **kwargs)
    else:
        marker = [(-d, -1), (d, 1)]
        kwargs = get_kwargs(marker)
        axs[0,0].plot(1, 1, transform=axs[0,0].transAxes, **kwargs)
        axs[0,1].plot(0, 1, transform=axs[0,1].transAxes, **kwargs)
        axs[1,0].plot(1, 0, transform=axs[1,0].transAxes, **kwargs)
        axs[1,1].plot(0, 0, transform=axs[1,1].transAxes, **kwargs)

        marker=[(-1, -d), (1, d)]
        kwargs = get_kwargs(marker)
        axs[0,0].plot(0, 0, transform=axs[0,0].transAxes, **kwargs)
        axs[0,1].plot(1, 0, transform=axs[0,1].transAxes, **kwargs)
        axs[1,0].plot(0, 1, transform=axs[1,0].transAxes, **kwargs)
        axs[1,1].plot(1, 1, transform=axs[1,1].transAxes, **kwargs)

    return axs.flat


def add_colorbar_map(plt, ax, color1='royalblue', color2='red', nslice=20,
                     vmin=.0, vmax=1., posx=.0, posy=.0, loc='right'):
    cm = mcolors.LinearSegmentedColormap.from_list('Custom', [color1, color2], nslice)
    cNorm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap=cm)
    plt.colorbar(scalarMap, ax=ax, location=loc, anchor=(posx,posy))
    return cm


def make_line_segments(x, y):
    points = np.column_stack((x, y))
    segments = np.concatenate([points[:-1, np.newaxis], points[1:, np.newaxis]], axis=1)
    return segments


def gradient_color_line(ax, x, y, weights, cmap, vmin=.0, vmax=1., label=''):
    segments = make_line_segments(x, y)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    line_segments = LineCollection(segments, cmap=cmap, norm=norm, color=cmap(norm(weights)),
                                   label=label)
    ax.add_collection(line_segments)
    return line_segments


def broadening(centers, heighs, width=0.0005, margin=0.05, method='gaussian',
               npoints=1001, xrange=None, **kwargs):
    r"""
    Generate broadened spectrum from stick spectrum.

    Args:
        centers (array_like): Positions of the stick spectrum peaks.
        heights (array_like): Heights of the stick spectrum peaks.
        width (float, optional): Broadening width parameter sigma. Default is 0.0005.
        margin (float, list or tuple, optional): Extra margin added to the min and max of centers for x-axis range. Default is 0.05.
        method (str, optional): Broadening method. Options are 'lorentzian', 'gaussian', and 'voigt'. Default is 'gaussian'.
        npoints (int, optional): Number of points in the output spectrum. Default is 1001.
        xrange (tuple, optional): Tuple specifying the x-axis range (min, max). If None, it is determined from centers and margin.
        gamma (float, optional): Lorentzian width parameter. Only used if method is 'voigt'.
        gamma_table (array_like, optional): Table of gamma values for each peak. Only used if method is 'voigt'.

    Returns:
        x (numpy.ndarray): x-axis values of the broadened spectrum.
        y (numpy.ndarray): y-axis values of the broadened spectrum.
    """
    if xrange:
        mi, ma = xrange
    else:
        if not isinstance(margin, (list, tuple)):
            margin = (margin, margin)
        mi, ma = np.min(centers)-margin[0], np.max(centers)+margin[1]

    x = np.linspace(mi, ma, npoints)
    y = 0.

    if method == 'lorentzian':
        for n in range(len(centers)):
            y += heighs[n] * width**2 / ((x-centers[n])**2 + width**2)
    elif method == 'gaussian':
        for n in range(len(centers)):
            y += np.sqrt(2*np.pi) * width * heighs[n] * norm.pdf(x, centers[n], width)
    elif method == 'voigt':
        from scipy.special import wofz
        gamma = kwargs.get('gamma', width)
        gamma_table = kwargs.get('gamma_table',
                                 [[820.310766e-9, 45.9188586e-9],
                                  [853.274140e-9, 48.0870613e-9],
                                  [889.567148e-9, 38.0147813e-9],
                                  [914.206437e-9, 42.7828469e-9],
                                  [945.514185e-9, 49.7859967e-9]])

        for n in range(len(centers)):
            gamma_pair = min(gamma_table, key=lambda g: abs(g[0]-centers[n]))
            z = ((x - centers[n]) + 1j*gamma_pair[1]*gamma) / (width * np.sqrt(2))
            y += (heighs[n] / (width * np.sqrt(2*np.pi))) * np.real(wofz(z))

    return x, y


def fit_val(positions, heighs, broaden):
    num = len(positions)
    x_min = positions[0] - 50
    if x_min < 0:
        x_min = 0
    x_max = positions[num-1] + 50
    if x_max > 4000:
        x_max = 4000
    ix = np.linspace(x_min, x_max, int(x_max-x_min))

    iy = 0.0
    for peak in range(num):
        iy += 2.51225 * broaden * heighs[peak] \
             * norm.pdf(ix, positions[peak], broaden)
    return (ix, iy)
