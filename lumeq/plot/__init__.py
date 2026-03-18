import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import ticker, animation
from matplotlib.collections import LineCollection

from .utils import brokenaxes, add_colorbar_map, gradient_color_line
from .utils import broadening, fit_val


def get_plot_colors(n):
    r"""
    Generate a list of distinct colors using matplotlib.

    Args:
        n (int): Number of colors to generate.

    Returns:
        list: Color strings.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for color generation.")

    prop_cycle = plt.rcParams['axes.prop_cycle']
    return prop_cycle.by_key()['color']
