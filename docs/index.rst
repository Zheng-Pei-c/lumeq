.. lumeq documentation master file, created by
   sphinx-quickstart on Sun Nov  9 11:14:03 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

lumeq documentation
===================

**lumeq** is a Python package for quantum chemistry developed by Zheng Pei at `CC-ATS group <https://sites.google.com/view/ccats-group/home>`_.
It provides tools for analyzing and manipulating wavefunctions, including functionalities for computing various properties and visualizations.
It also includes optimization algorithms for electronic structures based on Grassmann manifolds.
Besides, it offers embedding techniques for multiscale simulations of electronic states.
Molecular dynamics simulations are supported as well.
The other major feature considers spin model Hamiltonian simulations using various numerical methods
such as matrix product states, monte carlo, and quantum computing.
It dependes on the `PySCF <https://github.com/pyscf/pyscf>`_ package for quantum chemistry computations;
`qutip <https://qutip.org>`_ for spin model Hamiltonian simulations;
and `cirq <https://quantumai.google/cirq>`_ for quantum computing simulations.
Numpy, opt_einsum, and scipy are used for numerical matrix computations.


Check out the :doc:`usage` section for further information.

.. meta::
   :github_url: https://github.com/Zheng-Pei-c/lumeq


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   usage
   lumeq

.. note::
   This project is under active development.
   Please refer to the `GitHub repository <https://github.com/Zheng-Pei-c/lumeq>`_ for the latest updates and contributions.
