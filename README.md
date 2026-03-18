# LumeQ

Illuminating quantum in chemistry and materials with advanced computational tools.

LumeQ is a Python package for analyzing quantum chemical wavefunctions, built on top of [PySCF](https://github.com/pyscf/pyscf) and model Hamiltonian methods. This package is currently **under active development** and provides advanced tools for investigating electronic structure and dynamics. Key features include:

*   **Electronic Structure**:
    *   **MRSF DFT**: Mixed-Reference Spin-Flip DFT for robust description of ground and excited states, including conical intersections.
    *   **Polariton Chemistry**: Investigating light-matter interactions in cavity QED settings.
*   **Exciton and Molecular Dynamics**: Simulating time-dependent processes to understand energy transfer and structural evolution.
*   **Embedding Theory**: Utilizing Density Matrix Embedding Theory (DMET) and localized orbitals (e.g., Pipek-Mezey) to treat strong correlation in large systems.
*   **Molecular Property Analysis**: Calculating response properties, EPR parameters, intermolecular interactions (SAPT), and Energy Density.
*   **Spin Models**: Studying strongly correlated spin systems (Ising, Heisenberg) using Matrix Product States (DMRG), Monte Carlo simulations, and Quantum Computing algorithms (QAOA).
*   **utils**: Utility tools such as PySCF parser, unit conversion, Wick's theorem contractions.

For more details and API references, please visit the [lumeq](https://zheng-pei-c.github.io/lumeq/) webpage.



## Installation

You can install the package using `pip`.

```bash
git clone https://github.com/Zheng-Pei-c/lumeq.git
cd lumeq
pip install .
```

For development (editable mode):
```bash
pip install -e .
```

### Optional Dependencies
To include optional dependencies like `pyscf` (for electronic-structure workflows),
`spins` (for spin models, QAOA, and Wick-theorem utilities), or `cupy`
(for optional GPU performance monitoring):

```bash
pip install .[pyscf]
pip install .[spins]
pip install .[gpu]
# or combine them
pip install .[pyscf,spins,gpu]
```

### Manual Installation (PYTHONPATH)

Alternatively, you can add the package directory to your `PYTHONPATH`.

```bash
git clone https://github.com/Zheng-Pei-c/lumeq.git
export PYTHONPATH=$PYTHONPATH:/path/to/lumeq
```

**Note:** If you choose this method, you must manually install the required dependencies listed in [`pyproject.toml`](pyproject.toml) or refer to the [Dependencies](#dependencies) section below.



## Dependencies

Core dependencies:

-   [Python 3.x](https://www.python.org/)
-   [numpy](https://numpy.org/)
-   [scipy](https://scipy.org/)
-   [opt_einsum](https://github.com/dgasmith/opt_einsum)
-   [matplotlib](https://matplotlib.org/)
-   [psutil](https://github.com/giampaolo/psutil) (Required for memory usage monitoring)

Optional dependencies:

-   [PySCF](https://github.com/pyscf/pyscf) (`.[pyscf]`, used for electronic structure calculations)
-   [QuTiP](https://qutip.org/) (`.[spins]`)
-   [Cirq](https://quantumai.google/cirq) (`.[spins]`, used for quantum computing applications)
-   [Cirq-Google](https://pypi.org/project/cirq-google/) (`.[spins]`)
-   [SymPy](https://www.sympy.org/) (`.[spins]`)
-   [CuPy](https://cupy.dev/) (`.[gpu]`, optional, used for GPU performance monitoring)



## Usage Examples



### 1. Unit Conversion

The package provides a utility to convert between various physical units (time, length, energy, frequency, temperature, mass). See [`lumeq/utils/unit_conversion.py`](https://github.com/Zheng-Pei-c/lumeq/blob/main/lumeq/utils/unit_conversion.py).

```python
from lumeq.utils import convert_units

# Convert 100 fs to atomic units
t_au = convert_units(100, 'fs', 'au')
print(f"100 fs = {t_au} au")

# Convert 0.1 Hartree to eV
e_ev = convert_units(0.1, 'hartree', 'ev')
print(f"0.1 Hartree = {e_ev} eV")
```



### 2. Wick's Theorem Analysis

Wick's theorem is a powerful tool for analyzing quantum chemical wavefunctions.
The contraction of Wick's theorem for spin-quantized operators can be automated using the `sqo_evaluation` function in the `lumeq.utils.wick_contraction` module.
This example requires the optional spin dependencies:

```bash
pip install .[spins]
```

Here is an example of how to perform automated Wick's theorem contractions (adapted from [`samples/wick_operators.py`](https://github.com/Zheng-Pei-c/lumeq/blob/main/samples/wick_operators.py)).

```python
from lumeq.utils.wick_contraction import sqo_evaluation

# 1e Hamiltonian term
h1 = 'p_sigma^dagger q_tau'

# Open-shell excitation operators
Tsa = 's_alpha^dagger a_alpha' # bra side
Tbt = 'b_alpha^dagger t_alpha' # ket side
exceptions = [tuple(Tsa.split()), tuple(Tbt.split())]

# Evaluate contractions
sqo_evaluation(Tsa, h1, Tbt, exceptions=exceptions, title='Open-shell excited-state 1e term contractions', latex=True)
```



### 3. DMET Analysis

This example requires the electronic-structure dependencies:

```bash
pip install .[pyscf]
```

Here is an example of how to perform Density Matrix Embedding Theory (DMET) analysis on a water cluster (adapted from [`lumeq/embedding/fragment_entangle.py`](https://github.com/Zheng-Pei-c/lumeq/blob/main/lumeq/embedding/fragment_entangle.py)).

```python
from pyscf import gto, scf
from lumeq.embedding.fragment_entangle import get_embedding_system

# Define molecule (Water trimer example)
mol = gto.Mole()
mol.build(
    atom = """
       O         0.4183272099    0.1671038379    0.1010361156
       H         0.8784893276   -0.0368266484    0.9330933285
       H        -0.3195928737    0.7774121014    0.3045311682
       O         3.0208058979    0.6163509592   -0.7203724735
       H         3.3050376617    1.4762564664   -1.0295977027
       H         2.0477791789    0.6319690134   -0.7090745711
       O         2.5143150551   -0.2441947452    1.8660305097
       H         2.8954132119   -1.0661605274    2.1741344071
       H         3.0247679096    0.0221180670    1.0833062723
    """,
    basis = '6-311++g**',
    verbose=0
)

# Define fragments (atom indices)
frgm_idx = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

# Run SCF
mf = scf.RHF(mol)
mf.kernel()

# Perform DMET analysis
# Calculate embedding energy for the whole system from embedded fragments
get_embedding_system(mf, frgm_idx)
```



### 4. Molecular Dynamics

This example requires the electronic-structure dependencies:

```bash
pip install .[pyscf]
```

You can run molecular dynamics simulations using the `dynamics` module. See [`lumeq/dynamics/molecular_dynamics.py`](https://github.com/Zheng-Pei-c/lumeq/blob/main/lumeq/dynamics/molecular_dynamics.py) for more details.

```python
from lumeq.dynamics.molecular_dynamics import MolecularDynamics

# Setup MD parameters
key = {
    'atmsym': ['H', 'He'],
    'coordinate': [[0.0, 0.0, 0.0], [0.0, 0.0, 0.929352]],
    'velocity': [[0.0, 0.0, 0.0008], [0.0, 0.0, -0.0008]],
    'dt': 10,
    'total_time': 1000, # au
    'update_method': 'velocity_verlet'
}

# Setup Electronic Dynamics parameters
ed_key = {
    'functional': 'hf',
    'basis': 'sto-3g',
    'charge': 1,
    'ortho_method': 'lowdin'
}

# Initialize and run
md = MolecularDynamics(key, ed_key)
md.run_dynamics()
md.plot_time_variables(fig_name='md_trajectory.png')
```



## Modules

The package contains the following modules:

-   **`dynamics`**: Tools for simulating exciton dynamics and other time-dependent processes.
-   **`embedding`**: Methods for embedding calculations.
-   **`opt`**: Optimization routines for electronic structure calculations (states or molecular orbitals).
-   **`plot`**: Visualization tools for plotting results.
-   **`polariton`**: Analysis of polaritonic systems.
-   **`property`**: Calculation of molecular properties (e.g., EPR parameters, SAPT dispersion, energy density, etc.).
-   **`spins`**: Spin-related analysis based on various methods such as quantum Monte Carlo, density matrix renormalization group, quantum computing.

-   **`utils`**: General utility functions and PySCF parsers, for instance, PySCF parser, unit conversion, Wick's theorem contractions, printing matrix, etc.



## TODO

-   [ ] **Spins Module**: Method developments for the model Hamiltonian of spin systems (Monte Carlo, DMRG, Quantum Computing).
-   [ ] **Finite Temperature**: Add finite temperature support to electronic structure methods, dynamics, and spin models.
-   [ ] **Transport Module**: Implement `transport` module with system-bath interaction.
-   [ ] **Periodic Systems**: Add support for periodic system calculations.
-   [ ] **Grassmann Manifold Optimization**: Implement optimization algorithms on Grassmann manifolds for electronic states in `opt` module.
-   [ ] **Polariton Module**: Merge polariton to electronic structure module.
-   [ ] **Documentation**: Expand documentation for all modules.
-   [ ] **Tests**: Add comprehensive unit tests.



## Acknowledgements

This project utilizes AI-assisted coding tools, including [GitHub Copilot](https://github.com/github/copilot.vim), [Google Gemini](https://gemini.google.com/), and [ChatGPT](https://chatgpt.com/), for code generation and documentation.

The implementation details can be found in papers cited in the source files and my [personal notes](https://zhengpeic.github.io/notes/).


## License

[GNU General Public License v3.0](LICENSE)
