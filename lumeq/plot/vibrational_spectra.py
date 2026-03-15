from lumeq import sys, np
from lumeq.utils.pyscf_helper import *
from lumeq.utils import convert_units, print_matrix
from lumeq.plot import plt, fit_val

from scipy import signal, fftpack

from pyscf import lib
from pyscf.lib import logger
from pyscf.dft.rks import RKS
from pyscf.hessian import thermo

def plot_spectra(peak_centers, peak_intens, broaden, fig_name):
    ix, iy = fit_val(peak_centers, peak_intens, broaden)
    plt.plot(ix, iy)
    plt.xlim(800,3600)
    plt.ylim(0, 1200)
    plt.xticks([1000,1500,2000,2500,3000,3500], size=14)
    plt.yticks([0,400,800,1200], size=14)
    plt.xlabel("Frequency (cm$^{-1}$)",fontsize=16)
    plt.ylabel("Intensity",fontsize=16)
    plt.tight_layout()
    plt.savefig(fig_name)


def infrared(dip_dev, normal_mode):
    factor = 974.8802478
    if normal_mode.ndim == 3:
        # first index is mode
        normal_mode = normal_mode.reshape(-1, dip_dev.shape[0])

    trans_dip = np.einsum('qx,iq->ix', dip_dev, normal_mode)
    sir = np.einsum('ix,ix->i', trans_dip, trans_dip)
    return sir * factor


def get_dipole_dev(mf, hessobj, origin=None):
    if origin is None:
        if hasattr(mf, 'origin'): origin = mf.origin
        else: origin = np.zeros(3)

    mol = mf.mol
    natm = mol.natm
    atmlst = range(mol.natm)

    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    mocc = mo_coeff[:,mo_occ>0]
    nao = mo_coeff.shape[0]

    dipole = mol.intor('int1e_r', comp=3, hermi=0)
    dm = mf.make_rdm1()

    dipole_d1 = mol.intor('int1e_irp', comp=9, hermi=0).reshape(3,3,nao,nao).transpose(1,0,3,2)

    mo1 = lib.chkfile.load(hessobj.chkfile, 'scf_mo1')
    mo1 = {int(k): mo1[k] for k in mo1}

    g1 = np.zeros((natm, 3, 3)) # first 3 is derivative direction
    aoslices = mol.aoslice_by_atom()
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia, 2:]
        dm1 = np.einsum('xpi,qi->xpq', mo1[ia], mocc)

        g1[k] += np.einsum('ypq,xqp->xy', dipole, dm1)*2. # 2 for dm1
        g1[k] -= np.einsum('xypq,qp->xy', dipole_d1[:,:,p0:p1], dm[:,p0:p1]) # dipole_d1 has extra minus

    g1 *= -2. # electron dipole is negative # 2 for ket

    z = mol.atom_charges()
    for i in range(natm): # nuclear dipole derivative
        g1[i] += np.eye(3)*z[i]

    return g1.reshape(natm*3, 3)


def autocorrelation(arrays, dt, window='gaussian', domain='freq', direv=False):
    # C(t) = < \int A(\tau) B(t-\tau) \dd \tau >
    # Wiener-Khintchine theorem
    # first index is along time
    if arrays.ndim == 1:
        arrays = arrays.reshape(-1, 1)

    nstep = arrays.shape[0]
    if direv: # get time derivatives of the arrays
        arrays = np.gradient(arrays, edge_order=2, axis=0) / dt
    arrays -= np.mean(arrays, axis=0)
    norm = np.einsum('ix,ix->x', arrays, arrays)

    n = nstep*2 if nstep%2==0 else nstep*2-1
    correlation = np.zeros(arrays.shape)
    for i in np.where(norm>1e-8)[0]:
        #correlation[:,i] = signal.convolve(arrays[:,i], arrays[::-1,i], mode='full')[nstep-1:] / norm[i]
        tmp = np.zeros(n)
        tmp[nstep//2:nstep//2+nstep] = np.copy(arrays[:,i])
        correlation[:,i] = signal.convolve(tmp, arrays[::-1,i], mode='same')[-nstep:] / np.arange(nstep, 0, -1)

    #window = 'none'
    #if window == 'gaussian':
    #    sigma = 2. * np.sqrt(2. * np.log(2.))
    #    window = signal.gaussian(nstep, std=4000./sigma, sym=False)
    #elif hasattr(signal, window): # hann, hamming, blackmanharris
    #    window = getattr(signal, window)(nstep, sym=False)

    #if isinstance(window, np.ndarray):
    #    wf = window / np.sum(window) * nstep
    #    correlation = correlation * wf[:,None]

    if domain == 'time':
        return correlation
    elif domain == 'freq':
        return np.fft.fft2(correlation)[:nstep//2]


def fft_acf(arrays, dt, unit='au', scale_freq=True):
    nstep = arrays.shape[0]

    if unit != 's':
        dt = convert_units(dt, unit, 's')
    freq = np.fft.fftfreq(nstep, dt)[:nstep//2]

    #sigma = np.fft.fft2(arrays)[:nstep//2] / nstep
    sigma = fftpack.dct(arrays[:nstep//2], type=1, axis=0)
    sigma = np.mean(sigma, axis=1) # average

    if scale_freq:
        sigma *= freq**2

    freq = convert_units(freq, 'hz', 'cm-1')
    return freq, sigma


def smooth(x, window='hanning', window_len=11):
    """
    window: flat, hanning, hamming, bartlett, blackman
    """
    if window_len < 3:
        return x

    s = np.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]

    if window == 'flat':
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.'+window+'(window_len)')

    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[window_len//2-1:-window_len//2]


def cal_spectra(array, dt, window='gaussian', unit='au', scale_freq=True,
                direv=False, smoothing=True):
    correlation = autocorrelation(array, dt, window, 'time', direv)
    freq, sigma = fft_acf(correlation, dt, unit, scale_freq)

    if smoothing:
        sigma = smooth(sigma)

    return freq, sigma



if __name__ == '__main__':
    #infile = 'h2o.in'
    #parameters = parser(infile)

    #charge, spin, atom = parameters.get(section_names[0])[1:4]
    #functional, basis = get_rem_info(parameters.get(section_names[1]))[:2]
    #mol = build_molecule(atom, basis, charge, spin, verbose=0)

    hf = """
            H    0. 0. -0.459
            F    0. 0.  0.459
    """
    h2o = """
            H    1.6090032   -0.0510674    0.4424329
            O    0.8596350   -0.0510674   -0.1653507
            H    0.1102668   -0.0510674    0.4424329
    """
    co2 = """
            O    0. 0. 0.386715
            C    0. 0. 1.550000
            O    0. 0. 2.713285
    """
    atom = locals()[sys.argv[1]] if len(sys.argv) > 1 else hf

    functional = 'hf'
    mol = build_molecule(atom, 'sto-3g')

    mf = RKS(mol) # in coherent state
    mf.xc = functional
    mf.grids.prune = True

    e_tot = mf.kernel()
    hessobj = mf.Hessian()
    h = hessobj.kernel()
    print_matrix('hess:', h, 5, 1)

    results = thermo.harmonic_analysis(mol, h) # only molecular block
    print('freq_au:', results['freq_au'])
    print('freq_wavenumber:', results['freq_wavenumber'])
    print('force_const_dyne:', results['force_const_dyne'])
    print('reduced_mass:', results['reduced_mass'])

    dip_dev = get_dipole_dev(mf, hessobj)
    print_matrix('dip_dev:', dip_dev)
    sir = infrared(dip_dev, results['norm_mode'])
    print('infrared intensity:', sir)
