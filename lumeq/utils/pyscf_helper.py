from lumeq import sys, np
from lumeq.utils import print_matrix
from lumeq.utils.parser import parser, section_names

from pyscf import scf, tdscf, gto
from pyscf.lib import logger

#import qed

def build_atom(atmsym, coords):
    atom = ''
    for i in range(len(atmsym)):
        atom += str(atmsym[i]) + ' '
        for x in range(3):
            atom += str(coords[i,x]) + ' '
        atom += ';  '

    return atom


def build_molecule(atom, basis, charge=0, spin=0, unit='angstrom',
                          max_memory=60000, verbose=0):
    mol = gto.M(
        atom       = atom,
        unit       = unit,
        basis      = basis,
        spin       = spin,
        charge     = charge,
        max_memory = max_memory,
        verbose    = verbose
    )

    return mol


def get_jobtype(parameters):
    jobtype = 'scf'
    if 'cis_n_roots' in parameters.get(section_names[1]):
        jobtype = 'td'

    if 'force' in parameters.get(section_names[1]):
        jobtype += 'force'
    elif 'freq' in parameters.get(section_names[1]):
        jobtype += 'hess'

    if section_names[2] in parameters:
        jobtype += '_qed'

    return jobtype


def get_frgm_idx(parameters):
    frgm_idx = parameters.get(section_names[1])['impurity']
    if isinstance(frgm_idx, list):
        for i in range(len(frgm_idx)):
            at = frgm_idx[i].split('-')
            frgm_idx[i] = list(range(int(at[0])-1, int(at[1])))
    else:
        at = frgm_idx.split('-')
        frgm_idx = [list(range(int(at[0])-1, int(at[1])))] # need the outer bracket

    natm = len(np.ravel(parameters.get(section_names[0])[4]))
    assigned = np.concatenate(frgm_idx).tolist()
    if len(assigned) < natm:
        frgm_idx.append(list(set(range(natm)) - set(assigned)))

    #print('frgm_idx:', frgm_idx)
    return frgm_idx


def _get_center_of_mass(mol):
    mass = mol.atom_mass_list(isotope_avg=True)
    atom_coords = mol.atom_coords()
    mass_center = np.einsum('z,zx->x', mass, atom_coords) / mass.sum()
    return mass_center


def get_center_of_mass(mol, nfrag=1):
    if isinstance(mol, list):
        mass_center = [None]*nfrag
        for n in range(nfrag):
            mass_center[n] = _get_center_of_mass(mol[n])
        return np.array(mass_center)
    else:
        return _get_center_of_mass(mol)


def _run_pyscf_dft(charge, spin, atom, basis, functional, verbose=0, h=None,
                   scf_method='RKS'):
    mol = build_molecule(atom, basis, charge, spin, verbose=verbose)
    mf = getattr(scf, scf_method)(mol)
    if h:
        #h = h + mol.intor('cint1e_kin_sph') + mol.intor('cint1e_nuc_sph')
        mf.get_hcore = lambda *args: h
    mf.xc = functional
    mf.grids.prune = True
    etot = mf.kernel() # return total energy so that we don't need to calculate it again

    return mol, mf, etot


def run_pyscf_dft(charge, spin, atom, basis, functional, nfrag=1, verbose=0,
                  h=None, scf_method='RKS'):
    if isinstance(charge, list):
        mol, mf, etot = [None]*nfrag, [None]*nfrag, [None]*nfrag
        for n in range(nfrag):
            mol[n], mf[n], etot[n] = _run_pyscf_dft(charge[n], spin[n], atom[n],
                                                    basis, functional, verbose,
                                                    h, scf_method)

        return mol, mf, etot
    else:
        return _run_pyscf_dft(charge, spin, atom, basis, functional, verbose,
                              h, scf_method)


def _run_pyscf_tddft(mf, td_model, nroots, verbose=0):
    def rotation_strength(td, trans_dip=None, trans_mag_dip=None):
        if trans_dip is None: trans_dip = td.transition_dipole()
        if trans_mag_dip is None:
            #trans_mag_dip = td.trans_mag_dip
            trans_mag_dip = td.transition_magnetic_dipole()

        f = np.einsum('sx,sx->s', trans_dip, trans_mag_dip)
        return f

    td = getattr(tdscf, td_model)(mf)
    td.max_cycle = 600
    #td.max_space = 200

    if nroots > 0:
        td.kernel(nstates=nroots)
        if not td.converged.all():
            print('tddft is not converged:', td.converged)
        #try:
        #    td.converged.all()
        #    #print('TDDFT converged: ', td.converged)
        #    #print_matrix('Excited state energies (eV):\n', td.e * 27.2116, 6)
        #except Warning:
        #    #print('the %d-th job for TDDFT is not converged.' % (n+1))
        #    print('the job for TDDFT is not converged.')

    td.f_rotation = rotation_strength(td)

    if verbose >= 5:
        td.analyze(verbose)

    return td


def run_pyscf_tddft(mf, td_model, nroots, nfrag=1, verbose=0):
    if isinstance(mf, list):
        td = [None]*nfrag
        for n in range(nfrag):
            td[n] = _run_pyscf_tddft(mf[n], td_model, nroots, verbose=verbose)

        return td
    else:
        return _run_pyscf_tddft(mf, td_model, nroots, verbose)


def _run_pyscf_dft_tddft(charge, spin, atom, basis, functional, td_model, nroots,
                         verbose=0, debug=0, h=None, scf_method='RKS'):
    mol, mf, etot = _run_pyscf_dft(charge, spin, atom, basis, functional,
                                  verbose, h, scf_method)
    td = _run_pyscf_tddft(mf, td_model, nroots, verbose)
    return mol, mf, etot, td


# mainly for parallel execute
def run_pyscf_dft_tddft(charge, spin, atom, basis, functional, td_model, nroots,
                        nfrag=1, verbose=0, debug=0, h=None, scf_method='RKS'):
    if isinstance(charge, list):
        mol, mf, etot, td = [None]*nfrag, [None]*nfrag, [None]*nfrag, [None]*nfrag
        for n in range(nfrag):
            mol[n], mf[n], etot[n] = _run_pyscf_dft(charge[n], spin[n], atom[n],
                                                   basis, functional, verbose,
                                                   h, scf_method)
            td[n] = _run_pyscf_tddft(mf[n], td_model, nroots, verbose)

        if debug > 0:
            final_print_energy(td, nwidth=10, iprint=7)
            trans_dipole, trans_mag_dip, argmax = find_transition_dipole(td, nroots, nfrag)

        return mol, mf, etot, td
    else:
        return _run_pyscf_dft_tddft(charge, spin, atom, basis, functional,
                                    td_model, nroots, verbose, debug, h,
                                    scf_method)


def _run_pyscf_tdqed(mf, td, qed_model, cavity_model, key):
    cav_obj = getattr(qed, cavity_model)(mf, key)
    qed_td = getattr(qed, qed_model)(mf, td, cav_obj, key)
    qed_td.kernel()
    if not qed_td.converged.all():
        print('tdqed is not converged:', td.converged)
    #try:
    #    qed_td.converged.all()
    #    #e_lp, e_up = qed_td.e[:2]
    #    #print('e_lp:', e_lp, '  e_up:', e_up)
    #    #print_matrix('qed state energies(H):\n', qed_td.e)
    #except Warning:
    #    print('the job for qed-TDDFT is not converged.')

    return qed_td, cav_obj


def run_pyscf_tdqed(mf, td, qed_model, cavity_model, key, nfrag=1):
    if isinstance(mf, list):
        qed_td, cav_obj = [None]*nfrag, [None]*nfrag
        for n in range(nfrag):
            qed_td[n], cav_obj[n] = _run_pyscf_tdqed(mf[n], td[n], qed_model,
                                                     cavity_model, key)

        return qed_td, cav_obj
    else:
        return _run_pyscf_tdqed(mf, td, qed_model, cavity_model, key)


def _find_transition_dipole(td, nroots):
    trans_dipole = td.transition_dipole()
    trans_mag_dip = td.transition_magnetic_dipole()
    argmax = np.unravel_index(np.argmax(np.abs(trans_dipole), axis=None),
                              trans_dipole.shape)[0]
    print_matrix('trans_dipole:', trans_dipole, 10)
    print_matrix('trans_mag_dip:', trans_mag_dip, 10)
    return trans_dipole, trans_mag_dip, argmax


def find_transition_dipole(td, nroots, nfrag=1):
    if isinstance(td, list):
        trans_dipole, trans_mag_dip, argmax = [None]*nfrag, [None]*nfrag, [None]*nfrag
        for n in range(nfrag):
            trans_dipole[n], trans_mag_dip[n], argmax[n] = _find_transition_dipole(td[n], nroots)

        trans_dipole = np.reshape(trans_dipole, (nfrag, -1, 3))
        trans_mag_dip = np.reshape(trans_mag_dip, (nfrag, -1, 3))
        return trans_dipole, trans_mag_dip, argmax
    else:
        return _find_transition_dipole(td, nroots)


def find_oscillator_strength(td, nroots, nfrag=1):
    if isinstance(td, list):
        f_oscillator, f_rotation = [None]*nfrag, [None]*nfrag
        for n in range(nfrag):
            f_oscillator[n] = td[n].oscillator_strength()
            f_rotation[n] = td[n].f_rotation
        return np.array(f_oscillator), np.array(f_rotation)
    else:
        return td.oscillator_strength(), td.f_rotation


def final_print_energy(td, title='tddft', nwidth=6, iprint=0):
    if not isinstance(td, list): td = [td]

    if not isinstance(td[0].e, np.ndarray): return

    energy = []
    for n in range(len(td)):
        energy.append(td[n].e)
    energy = np.reshape(energy, (len(td), -1))

    if iprint > 0:
        print_matrix(title+' energy:', energy, nwidth)

    return energy


def get_basis_info(mol):
    nbas = mol.nao_nr()
    nocc = mol.nelectron // 2 # assume closed-shell even electrons
    nvir = nbas - nocc
    nov  = nocc * nvir

    return [nbas, nocc, nvir, nov]


def get_rem_info(rem_keys):
    for key in rem_keys:
        print(key + ' = ', end=' ')
        key = rem_keys.get(key)
        print(key)

    functional = rem_keys.get('method')
    basis = rem_keys.get('basis')

    unrestricted = rem_keys.get('unrestricted', 0)
    if unrestricted in {0, 'false', 'FALSE'}:
        scf_method = 'RKS'
    elif unrestricted in {'2', 'g', 'general'}:
        scf_method = 'GKS'
    else:
        scf_method = 'UKS'

    nroots = rem_keys.get('cis_n_roots', 0)
    # 0 for rpa if it is ungiven. But it's working! Still None
    rpa = rem_keys.get('rpa', 0)
    td_model = 'TDDFT' if rpa == 2 else 'TDA'

    verbose = rem_keys.get('verbose', 1)
    debug   = rem_keys.get('debug', 0)

    return functional, basis, nroots, td_model, verbose, debug, scf_method


def get_photon_info(photon_key):
    key = photon_key.copy()

    if isinstance(key.get('cavity_model', 'JC'), list): # support many models
        key['cavity_model'] = [x.capitalize() if x.upper()=='RABI' else x.upper() for x in key['cavity_model']]
    else:
        x = key.get('cavity_model')
        key['cavity_model'] = x.capitalize() if x.upper()=='RABI' else x.upper()
    if key.get('cavity_freq', None):
        key['cavity_freq'] = np.array([key.get('cavity_freq')])
    else:
        raise TypeError('need cavity frequency')
    key['uniform_field'] = bool(key.get('uniform_field', True))
    key.setdefault('efield_file', 'efield')
    #if key.get('cavity_mode', None):
    cavity_mode = key.get('cavity_mode', None)
    if isinstance(cavity_mode, list) or isinstance(cavity_mode, np.ndarray):
        key['cavity_mode'] = np.array(key['cavity_mode']).reshape(3, -1)
    else:
        if key['uniform_field']:
            raise TypeError('need cavity mode with uniform field')
        else:
            key['cavity_mode'] = np.ones((3, 1)) # artificial array

    key.setdefault('freq_window', [-0.05, 0.05])
    key['solver_algorithm'] = key.get('solver_algorithm', 'davidson_qr').lower()
    key['solver_conv_prop'] = key.get('solver_conv_prop', 'norm').lower()
    key['target_states'] = key.get('target_states', 'polariton').lower()
    key.setdefault('nstates', 4)
    #key.setdefault('solver_nvecs', 4)
    key['solver_conv_thresh'] = pow(10, -key.get('solver_conv_thresh', 8))
    key.setdefault('rpa', 0)
    key['qed_model'] = 'RPA' if key['rpa'] == 2 else 'TDA'
    key.setdefault('resonance_state', None)
    key.setdefault('verbose', 0)
    key.setdefault('debug', 0)

    key.setdefault('save_data', 0)
    key.setdefault('max_cycle', 50)
    key.setdefault('tolerance', 1e-9)
    key.setdefault('level_shift', 1e-2)

    print('qed_cavity_model: %s/%s' % (key['qed_model'], key['cavity_model']))
    print('cavity_mode: ', key['cavity_mode'])
    print('cavity_freq: ', key['cavity_freq'])

    return key


def justify_photon_info(td, nroots, nstate='max_dipole', func='average', nwidth=10):
    energy = final_print_energy(td, nwidth=nwidth)
    if nstate == 'max_dipole':
        trans_dipole, trans_mag_dip, argmax = find_transition_dipole(td, nroots)
        argmax0 = argmax[0] if isinstance(argmax, np.ndarray) else argmax
        print_matrix('max tddft energy:', energy[:, argmax0].T, nwidth=nwidth)
    elif isinstance(nstate, int):
        argmax0 = nstate - 1

    if func == 'average':
        freq = getattr(np, func)(energy[:, argmax0])
        print('change applied photon energy to a more suitable one as:', freq)
    elif 'fwhm' in func: # full width at half maximum
        data = func.split('-')
        if len(data) > 1:
            func, factor = data[0], float(data[1])
        else: factor = 1.
        std = np.std(energy[:, argmax0])
        ave = np.average(energy[:, argmax0])
        if func[-2:] == '_m' or func[-2:] == '_l':
            freq = ave - 2. /factor * np.sqrt(2.*np.log(2.)) * std
        else:
            freq = ave + 2. /factor * np.sqrt(2.*np.log(2.)) * std
        print('change applied photon energy to a more suitable one as:', freq)


    return argmax0, np.asarray([freq])


def run_pyscf_final(parameters):
    cpu0 = (logger.process_clock(), logger.perf_counter())
    log = logger.new_logger(verbose=5)

    nfrag, charge, spin, atom = parameters.get(section_names[0])[:4]
    functional, basis, nroots, td_model, verbose, debug, scf_method = \
                get_rem_info(parameters.get(section_names[1]))


    results = {}

    jobtype = get_jobtype(parameters)
    if jobtype == 'scf':
        mol, mf, etot = run_pyscf_dft(charge, spin, atom, basis, functional,
                                      nfrag, verbose, scf_method=scf_method)
        results['mol'] = mol
        results['mf']  = mf
        print_matrix('scf energy:', [etot])
    elif 'td' in jobtype:
        mol, mf, etot, td = run_pyscf_dft_tddft(charge, spin, atom, basis, functional,
                                                td_model, nroots, nfrag, verbose, debug,
                                                scf_method=scf_method)
        results['mol'] = mol
        results['mf']  = mf
        results['td']  = td
        print_matrix('scf energy:', [etot])
        final_print_energy(td, nwidth=10, iprint=1)

    if 'td_qed' in jobtype:
        mol0 = mol[0] if isinstance(mol, list) else mol
        nov = get_basis_info(mol0)[-1] # assume identical molecules
        if nroots > nov: # fix the nroots if necessary
            nroots = nov
            if isinstance(td, list):
                for n in range(len(td)): td[n].nroots = nov
            else: td.nroots = nov

        key = get_photon_info(parameters.get(section_names[2]))

        cavity_model = key['cavity_model']
        if not isinstance(cavity_model, list): cavity_model = [cavity_model]
        n_model = len(cavity_model)
        qed_td, cav_obj = [None]*n_model, [None]*n_model
        for i in range(len(cavity_model)):
            qed_td[i], cav_obj[i] = run_pyscf_tdqed(mf, td, key['qed_model'],
                                                    cavity_model[i], key, nfrag)

        for i in range(n_model):
            final_print_energy(qed_td[i], cavity_model[i]+' qed-tddft', 10, iprint=1)

        results['qed_td'] = qed_td
        results['cav_obj'] = cav_obj

    log.timer('pyscf running time', *cpu0)
    return results



if __name__ == '__main__':
    infile = 'water.in'
    if len(sys.argv) >= 2: infile = sys.argv[1]
    parameters = parser(infile)
    results = run_pyscf_final(parameters)
