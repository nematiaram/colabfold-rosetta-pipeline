#!/usr/bin/env python3
"""Run Rosetta per-residue solvent exposure (sphere or cone neighbor count)."""
import argparse
import os
import re
import subprocess
from pathlib import Path

import pandas as pd

ROSETTA_BIN_DEFAULT = os.environ.get(
    "ROSETTA_BIN",
    "per_residue_solvent_exposure.linuxgccrelease",
)

AA1_TO3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
    "U": "SEC", "O": "PYL",
}
AA3 = set(AA1_TO3.values())


def _first_int_token(toks):
    for t in toks:
        try:
            return int(t)
        except ValueError:
            continue
    return None


def _last_float_token(toks):
    for t in reversed(toks):
        try:
            return float(t)
        except ValueError:
            continue
    return None


def _parse_resname_from_tokens(toks):
    for t in toks:
        u = t.upper()
        if u in AA3:
            return u
    for t in toks:
        if len(t) == 1 and t.upper() in AA1_TO3:
            return AA1_TO3[t.upper()]
    return "UNK"


def _parse_chain_from_tokens(toks, default_chain: str) -> str:
    for t in toks:
        m = re.match(r"(?i)^chain[:=]([A-Za-z0-9])$", t)
        if m:
            return m.group(1)
    for i, t in enumerate(toks[:-1]):
        if t.lower() == "chain" and len(toks[i + 1]) == 1:
            return toks[i + 1]
    if len(toks) >= 4:
        if len(toks[0]) == 1 and toks[0].upper() in AA1_TO3:
            try:
                int(toks[1])
                if len(toks[2]) == 1 and toks[2].isalnum():
                    return toks[2]
            except ValueError:
                pass
    return default_chain


def parse_nc_file(path: str, col_label: str, default_chain: str = "A") -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("residue"):
                continue
            toks = line.split()
            if len(toks) < 2:
                continue
            resnum = _first_int_token(toks)
            nc_val = _last_float_token(toks)
            if resnum is None or nc_val is None:
                continue
            resname = _parse_resname_from_tokens(toks)
            chain = _parse_chain_from_tokens(toks, default_chain=default_chain)
            rows.append({"chain": chain, "resnum": resnum, "resname": resname, col_label: nc_val})
    if not rows:
        raise RuntimeError(f"No NC rows parsed from {path}")
    df = pd.DataFrame(rows)
    return df.groupby(["chain", "resnum", "resname"], as_index=False).agg({col_label: "mean"})


def read_nc_table(path, col_label, default_chain: str = "A"):
    try:
        df = pd.read_csv(path, sep=r"\s+", engine="python")
        required = {"Residue_Type", "Residue_Number", "Neighbor_Count"}
        if required.issubset(set(df.columns)):
            out = df[["Residue_Type", "Residue_Number", "Neighbor_Count"]].copy()
            out = out.rename(columns={
                "Residue_Type": "resname",
                "Residue_Number": "resnum",
                "Neighbor_Count": col_label,
            })
            out["chain"] = default_chain
            return out[["chain", "resnum", "resname", col_label]]
    except Exception:
        pass
    return parse_nc_file(path, col_label, default_chain=default_chain)


def merge_nc(uid: str, out_dir: Path, method: str, default_chain: str = "A") -> Path:
    files = [
        out_dir / f"{uid}_rep_cluster{i}_neighbor_count_{method}.out"
        for i in (1, 2, 3)
    ]
    for fp in files:
        if not fp.is_file():
            raise FileNotFoundError(f"Missing NC file: {fp}")

    df1 = read_nc_table(files[0], "nc_rep1", default_chain=default_chain)
    df2 = read_nc_table(files[1], "nc_rep2", default_chain=default_chain)
    df3 = read_nc_table(files[2], "nc_rep3", default_chain=default_chain)

    merged = df1.merge(df2, on=["chain", "resnum"], how="inner", suffixes=("", "_b"))
    merged = merged.merge(df3, on=["chain", "resnum"], how="inner", suffixes=("", "_c"))
    for suffix in ("_b", "_c"):
        col = f"resname{suffix}"
        if col in merged.columns:
            merged["resname"] = merged["resname"].where(
                merged["resname"].ne("UNK"), merged[col].fillna(merged["resname"])
            )
            merged = merged.drop(columns=[col])

    merged = merged[["chain", "resnum", "resname", "nc_rep1", "nc_rep2", "nc_rep3"]]
    merged = merged.sort_values(["chain", "resnum"]).reset_index(drop=True)
    out_path = out_dir / f"{uid}_rosetta_nc.tsv"
    merged.to_csv(out_path, sep="\t", index=False)
    return out_path


def main():
    ap = argparse.ArgumentParser(
        description="Run Rosetta neighbor count (NC) on representative models."
    )
    ap.add_argument("--uniprot", required=True)
    ap.add_argument("--rep-info", required=True,
                    help="TSV with rep_id and pdb_path (from script 02).")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rosetta-bin", default=ROSETTA_BIN_DEFAULT)
    ap.add_argument("--method", choices=["sphere", "cone"], default="cone",
                    help="Rosetta solvent_exposure method (default: cone).")
    ap.add_argument("--dist-midpoint", type=float, default=9.0)
    ap.add_argument("--dist-steepness", type=float, default=1.0)
    ap.add_argument("--default-chain", default="A",
                    help="Default chain ID when NC output lacks chain.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rep_info = Path(args.rep_info)
    if not rep_info.is_file():
        raise FileNotFoundError(f"Missing rep_info TSV: {rep_info}")

    rows = []
    with rep_info.open() as f:
        header = f.readline().strip().split("\t")
        if "rep_id" not in header or "pdb_path" not in header:
            raise RuntimeError("rep_info must contain columns rep_id and pdb_path")
        idx_rep = header.index("rep_id")
        idx_pdb = header.index("pdb_path")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) <= max(idx_rep, idx_pdb):
                continue
            rows.append((parts[idx_rep], parts[idx_pdb]))

    db = os.environ.get("ROSETTA3_DB") or os.environ.get("ROSETTA_DATABASE")
    for rep_id, pdb_path in rows:
        base = f"{args.uniprot}_{rep_id}"
        out_path = out_dir / f"{base}_neighbor_count_{args.method}.out"
        cmd = [
            args.rosetta_bin,
            "-in:file:s", str(pdb_path),
            "-solvent_exposure:method", args.method,
            "-dist_midpoint", str(args.dist_midpoint),
            "-dist_steepness", str(args.dist_steepness),
            "-out:file:o", str(out_path),
        ]
        if db:
            cmd[1:1] = ["-database", db]
        print("[RUN] " + " ".join(cmd))
        subprocess.run(cmd, check=True)

    nc_path = merge_nc(args.uniprot, out_dir, args.method, default_chain=args.default_chain)
    print(f"[DONE] method={args.method}  merged={nc_path}")


if __name__ == "__main__":
    main()
