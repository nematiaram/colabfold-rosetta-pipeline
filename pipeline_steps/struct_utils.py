#!/usr/bin/env python3
"""Utility functions for simple PDB parsing and RMSD.

This module intentionally avoids heavy dependencies.
"""

from typing import Dict, Tuple, List
import numpy as np

# 1-letter <-> 3-letter amino acid maps (standard set)
AA1_TO3: Dict[str, str] = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN",
    "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS",
    "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP",
    "Y": "TYR", "V": "VAL", "U": "SEC", "O": "PYL",
}
AA3_TO1: Dict[str, str] = {v: k for k, v in AA1_TO3.items()}


def get_ca_coords(pdb_path: str) -> np.ndarray:
    """Return Nx3 array of CA coordinates from a PDB."""
    coords: List[List[float]] = []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            coords.append([x, y, z])
    return np.array(coords, dtype=float)


def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """Kabsch RMSD between two Nx3 arrays P and Q."""
    if P.shape != Q.shape:
        raise ValueError(f"Shape mismatch in RMSD: P{P.shape} vs Q{Q.shape}")
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("Inputs must be Nx3 arrays")

    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    C = Pc.T @ Qc
    V, S, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])
    U = V @ D @ Wt
    P_rot = Pc @ U
    diff = P_rot - Qc
    return float(np.sqrt((diff * diff).sum() / P.shape[0]))


def parse_mean_plddt_from_pdb(pdb_path: str) -> float:
    """Mean pLDDT from B-factor column for CA atoms."""
    vals: List[float] = []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            b = float(line[60:66])
            vals.append(b)
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def pdb_residue_table(
    pdb_path: str,
    atom_name: str = "CA",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract per-residue (chain, resnum, resname) from a PDB."""
    chains: List[str] = []
    resnums: List[int] = []
    resnames: List[str] = []

    seen = set()
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != atom_name:
                continue
            chain = line[21].strip() or "-"
            resnum = int(line[22:26])
            resname = line[17:20].strip().upper() or "UNK"
            key = (chain, resnum)
            if key in seen:
                continue
            seen.add(key)
            chains.append(chain)
            resnums.append(resnum)
            resnames.append(resname)

    return (
        np.array(chains, dtype=object),
        np.array(resnums, dtype=int),
        np.array(resnames, dtype=object),
    )
