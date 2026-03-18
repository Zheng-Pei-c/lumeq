import os, sys
import numpy as np
import scipy
import itertools

from opt_einsum import contract
np.einsum = contract # replace numpy einsum with opt_einsum version

import lumeq
#from lumeq import utils

