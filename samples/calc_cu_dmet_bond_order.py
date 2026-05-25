#!/usr/bin/env python3
"""Calculate Cu-cluster DMET bond orders from XYZ files.

The script treats all Cu atoms as one fragment and all non-Cu atoms as the
second fragment. The reported DMET bond order is the sum of squared singular
values of the off-diagonal localized-orbital density block between the two
fragments. Defaults are PBE/def2-SVP with a LANL2DZ ECP on Cu; pass
``--ecp none`` for an all-electron def2-SVP calculation.
"""

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from pyscf import dft, gto
from pyscf.data import elements

from lumeq.property.bond_order import bond_order


def read_xyz(path):
    with open(path) as handle:
        lines = [line.strip() for line in handle if line.strip()]

    natm = int(lines[0])
    records = []
    for line in lines[2:2 + natm]:
        fields = line.split()
        records.append((fields[0], tuple(float(x) for x in fields[1:4])))
    if len(records) != natm:
        raise ValueError("%s has %d atoms, expected %d" % (path, len(records), natm))
    return records


def atom_string(records):
    return "\n".join(
        "%-2s %18.10f %18.10f %18.10f" % (symbol, xyz[0], xyz[1], xyz[2])
        for symbol, xyz in records
    )


def electron_count(records, charge):
    return sum(elements.charge(symbol) for symbol, _ in records) - charge


def metal_ecp(args):
    if args.ecp and args.ecp.lower() not in ("none", "no", "false"):
        return {args.metal: args.ecp}
    return {}


def build_mol(records, args):
    nelec = electron_count(records, args.charge)
    spin = args.spin
    if spin is None:
        spin = nelec % 2

    return gto.M(
        atom=atom_string(records),
        basis=args.basis,
        ecp=metal_ecp(args),
        charge=args.charge,
        spin=spin,
        unit="Angstrom",
        verbose=args.verbose,
    )


def build_mf(records, args):
    mol = build_mol(records, args)

    if mol.spin == 0 and not args.force_uks:
        mf = dft.RKS(mol)
    else:
        mf = dft.UKS(mol)
    mf.xc = args.xc
    mf.max_cycle = args.max_cycle
    mf.conv_tol = args.conv_tol
    mf.kernel()
    return mf


def fragment_indices(records, metal):
    metal_idx = [i for i, (symbol, _) in enumerate(records) if symbol == metal]
    rest_idx = [i for i, (symbol, _) in enumerate(records) if symbol != metal]
    if not metal_idx:
        raise ValueError("No %s atoms found" % metal)
    if not rest_idx:
        raise ValueError("No non-%s atoms found" % metal)
    return [metal_idx, rest_idx]


def calculate_file(path, args):
    records = read_xyz(path)
    fragments = fragment_indices(records, args.metal)
    mf = build_mf(records, args)
    dmet_bo, singular_values = bond_order(
        method="dmet",
        mf=mf,
        fragments=fragments,
        lo_method=args.lo_method,
        min_weight=args.min_weight,
        return_singular_values=True,
    )

    if isinstance(singular_values, list):
        s_alpha = singular_values[0][(0, 1)]
        s_beta = singular_values[1][(0, 1)]
    else:
        s_alpha = singular_values[(0, 1)]
        s_beta = []

    return {
        "file": path.name,
        "natm": len(records),
        "n_metal": len(fragments[0]),
        "n_rest": len(fragments[1]),
        "nelectron": mf.mol.nelectron,
        "spin": mf.mol.spin,
        "scf": mf.__class__.__name__,
        "converged": mf.converged,
        "energy_hartree": mf.e_tot,
        "dmet_bond_order": dmet_bo[0, 1],
        "singular_values_alpha": " ".join("%.10g" % x for x in s_alpha),
        "singular_values_beta": " ".join("%.10g" % x for x in s_beta),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate DMET bond order between Cu and adsorbate fragments."
    )
    parser.add_argument(
        "--xyz-dir",
        default="/Users/zheng/Downloads/SI_CLUSTER_XYZ",
        help="Directory containing CLUSTER_Cu_*.xyz files.",
    )
    parser.add_argument("--pattern", default="CLUSTER_Cu_*.xyz")
    parser.add_argument("--output", default="cu_dmet_bond_order.csv")
    parser.add_argument("--metal", default="Cu")
    parser.add_argument("--basis", default="def2-svp")
    parser.add_argument(
        "--ecp",
        default="lanl2dz",
        help="Molecular ECP for the metal atoms. Use 'none' to disable.",
    )
    parser.add_argument("--xc", default="pbe")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument(
        "--spin",
        type=int,
        default=None,
        help="2S value. Defaults to electron-count parity: 0 for even, 1 for odd.",
    )
    parser.add_argument("--force-uks", action="store_true")
    parser.add_argument("--lo-method", default="lowdin")
    parser.add_argument("--min-weight", type=float, default=0.8)
    parser.add_argument("--max-cycle", type=int, default=100)
    parser.add_argument("--conv-tol", type=float, default=1e-8)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list selected files and fragment sizes; do not run SCF.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    xyz_paths = sorted(Path(args.xyz_dir).glob(args.pattern))
    if args.max_files is not None:
        xyz_paths = xyz_paths[:args.max_files]
    if not xyz_paths:
        raise RuntimeError("No XYZ files matched %s in %s" % (args.pattern, args.xyz_dir))

    if args.dry_run:
        for path in xyz_paths:
            records = read_xyz(path)
            fragments = fragment_indices(records, args.metal)
            mol = build_mol(records, args)
            print(
                "%s: natm=%d %s=%d rest=%d nelectron=%d spin=%d"
                % (
                    path.name,
                    len(records),
                    args.metal,
                    len(fragments[0]),
                    len(fragments[1]),
                    mol.nelectron,
                    mol.spin,
                )
            )
        return

    rows = []
    for path in xyz_paths:
        print("Running %s" % path.name, flush=True)
        rows.append(calculate_file(path, args))
        print(
            "  DMET bond order = %.10f, converged = %s"
            % (rows[-1]["dmet_bond_order"], rows[-1]["converged"]),
            flush=True,
        )

    output = Path(args.output)
    with open(output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Wrote %s" % output)


if __name__ == "__main__":
    main()
