#!/usr/bin/env python3
import argparse
import glob
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Paper methods: discard models dominated by nonregular SS (C/S/T/unassigned).
COIL_LIKE = frozenset({"C", "S", "T", " ", "-", ""})

# DSSP secondary-structure filters are meaningless on tiny peptides: nearly every
# residue is coil-like, so the paper 0.60 cutoff drops the entire ensemble and
# aborts the job. Auto-disable below this CA count (ROSIE short smoke tests).
MIN_RESIDUES_FOR_COIL_FILTER = 30


def find_dssp_bin() -> str:
    for name in ("mkdssp", "dssp"):
        path = shutil.which(name)
        if path:
            return path
    raise RuntimeError(
        "DSSP binary not found (tried mkdssp, dssp). "
        "Install the 'dssp' package in the image, or set --max-coil-fraction 1.0 to disable."
    )


def write_dssp_friendly_pdb(src: str, dst: Path) -> None:
    """Rewrite a ColabFold PDB into a form DSSP 4 will accept.

    ColabFold unrelaxed PDBs typically start with ``MODEL`` and omit ``CRYST1``.
    DSSP 4 / libcifpp then fails with: Expected record CRYST1 but found MODEL.
    We emit a dummy CRYST1 and keep only ATOM/HETATM records (first MODEL only).
    """
    atoms = []
    in_model = False
    saw_model = False
    with open(src, encoding="utf-8", errors="replace") as f:
        for line in f:
            key = line[:6]
            if key.startswith("MODEL"):
                if saw_model:
                    break  # only first model
                saw_model = True
                in_model = True
                continue
            if key.startswith("ENDMDL"):
                if in_model:
                    break
                continue
            if key.startswith("ATOM  ") or key.startswith("HETATM"):
                atoms.append(line if line.endswith("\n") else line + "\n")
            elif key.startswith("TER"):
                atoms.append(line if line.endswith("\n") else line + "\n")

    if not atoms:
        raise RuntimeError(f"No ATOM/HETATM records in {src}")

    with open(dst, "w", encoding="utf-8") as out:
        out.write(
            "CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1\n"
        )
        out.writelines(atoms)
        out.write("END\n")


def _maybe_set_libcifpp_data_dir() -> None:
    """Point DSSP 4 at components.cif when conda/local installs need it."""
    if os.environ.get("LIBCIFPP_DATA_DIR"):
        return
    candidates = [
        Path("/usr/share/libcifpp"),
        Path("/usr/local/share/libcifpp"),
    ]
    # Common conda-forge layout next to the dssp binary.
    dssp = shutil.which("mkdssp") or shutil.which("dssp")
    if dssp:
        prefix = Path(dssp).resolve().parent.parent
        candidates.extend([
            prefix / "share" / "libcifpp",
            prefix / "share" / "dssp",
        ])
    for base in candidates:
        cif = base / "components.cif"
        if cif.is_file():
            os.environ["LIBCIFPP_DATA_DIR"] = str(base)
            return


def coil_fraction(pdb_path: str, dssp_bin: Optional[str] = None) -> float:
    """Fraction of residues with coil-like / nonregular DSSP assignment.

    Handles both classic DSSP text (CMBI 2.x/3.x) and mmCIF (DSSP 4+, which
    defaults to mmCIF). Counts C, S, T, blank, and '-' as nonregular per the
    paper methods.

    ColabFold PDBs are rewritten with a dummy CRYST1 (and without MODEL wrappers)
    before DSSP runs — required for DSSP 4.
    """
    bin_path = dssp_bin or find_dssp_bin()
    pdb_path = str(pdb_path)
    _maybe_set_libcifpp_data_dir()

    with tempfile.TemporaryDirectory(prefix="dssp_") as tmp:
        tmp_dir = Path(tmp)
        friendly = tmp_dir / "input.pdb"
        out_path = tmp_dir / "out.dssp"
        write_dssp_friendly_pdb(pdb_path, friendly)

        # 1. DSSP 4 with the flag that forces the classic text format we prefer.
        # 2. Default modern invocation (DSSP 4 will emit mmCIF here; the parser
        #    detects and handles it).
        # 3. Legacy CMBI -i/-o form for older binaries.
        attempts = [
            [bin_path, "--output-format", "dssp", str(friendly), str(out_path)],
            [bin_path, str(friendly), str(out_path)],
            [bin_path, "-i", str(friendly), "-o", str(out_path)],
        ]
        errors = []
        text = None
        for cmd in attempts:
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                    env=os.environ.copy(),
                )
            except (OSError, subprocess.SubprocessError) as e:
                errors.append("%s -> %s" % (" ".join(cmd), e))
                continue
            if proc.returncode == 0 and out_path.is_file() and out_path.stat().st_size > 0:
                text = out_path.read_text(encoding="utf-8", errors="replace")
                break
            detail = (proc.stderr or proc.stdout or "").strip().replace("\n", " | ")[:300]
            errors.append("rc=%s cmd=%s err=%s" % (proc.returncode, " ".join(cmd), detail))
            try:
                out_path.unlink()
            except OSError:
                pass
        if text is None:
            raise RuntimeError(
                "DSSP failed on %s after sanitize: %s" % (pdb_path, " || ".join(errors))
            )

    return _parse_dssp_coil_fraction(text)


def _parse_dssp_coil_fraction(text: str) -> float:
    """Coil-like fraction from DSSP output, format auto-detected.

    Classic text (CMBI 2.x/3.x): STRUCTURE column at index 16, one row per residue.
    mmCIF (DSSP 4+): `_dssp_struct_summary.secondary_structure` inside a `loop_`.
    """
    if _looks_like_mmcif(text):
        ss_codes = _parse_mmcif_dssp_ss(text)
    else:
        ss_codes = _parse_classic_dssp_ss(text)

    if not ss_codes:
        raise RuntimeError("No DSSP residue records parsed")

    n_coil = sum(1 for ss in ss_codes if ss in COIL_LIKE)
    return float(n_coil) / float(len(ss_codes))


def _looks_like_mmcif(text: str) -> bool:
    head = text.lstrip()[:4096]
    if head.startswith("data_"):
        return True
    if "_dssp_struct_summary" in head:
        return True
    return "loop_" in head and "_dssp_" in head and "_atom_site" not in head[:200]


def _parse_classic_dssp_ss(text: str) -> list:
    lines = text.splitlines()
    start = 0
    for i, line in enumerate(lines):
        if "  #  RESIDUE" in line or line.lstrip().startswith("#  RESIDUE"):
            start = i + 1
            break
    # If no banner, scan all non-empty lines of sufficient length (start stays 0).

    ss_codes = []
    for line in lines[start:]:
        if len(line) < 17:
            continue
        # Classic DSSP: AA at column 14 (index 13), STRUCTURE at column 17 (index 16).
        aa = line[13]
        if aa == "!":
            continue  # chain break
        if not aa.isalpha():
            continue
        ss_codes.append(line[16])
    return ss_codes


def _parse_mmcif_dssp_ss(text: str) -> list:
    """Extract SS codes from DSSP 4 mmCIF output.

    Walks each `loop_` block, looks for the `.secondary_structure` column, and
    reads its value from every data row of that loop. `.` and `?` (mmCIF null /
    unknown) both count as blank, matching the classic-format behavior.
    """
    lines = text.splitlines()
    ss_codes = []
    i, n = 0, len(lines)
    while i < n:
        if lines[i].strip() != "loop_":
            i += 1
            continue
        i += 1
        headers = []
        while i < n and lines[i].lstrip().startswith("_"):
            headers.append(lines[i].strip())
            i += 1
        ss_idx = next(
            (j for j, h in enumerate(headers) if h.endswith(".secondary_structure")),
            None,
        )
        if ss_idx is None:
            # Skip data rows of this loop.
            while i < n:
                s = lines[i].strip()
                if not s or s.startswith("#") or s == "loop_" or s.startswith("_") or s.startswith("data_"):
                    break
                i += 1
            continue
        while i < n:
            s = lines[i].strip()
            if not s or s.startswith("#") or s == "loop_" or s.startswith("_") or s.startswith("data_"):
                break
            fields = _mmcif_split(s)
            if len(fields) > ss_idx:
                code = fields[ss_idx]
                if code in (".", "?"):
                    ss_codes.append(" ")
                else:
                    ss_codes.append(code[:1] or " ")
            i += 1
    return ss_codes


def _mmcif_split(row: str) -> list:
    """Whitespace split respecting single- and double-quoted fields."""
    fields, i, n = [], 0, len(row)
    while i < n:
        ch = row[i]
        if ch.isspace():
            i += 1
            continue
        if ch in "'\"":
            quote = ch
            i += 1
            start = i
            while i < n and row[i] != quote:
                i += 1
            fields.append(row[start:i])
            i += 1
        else:
            start = i
            while i < n and not row[i].isspace():
                i += 1
            fields.append(row[start:i])
    return fields


def get_ca_coords(pdb_path: str) -> np.ndarray:
    coords = []
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
    arr = np.array(coords, dtype=float)
    if arr.size == 0:
        raise RuntimeError(f"No CA atoms in {pdb_path}")
    return arr


def parse_mean_plddt_from_pdb(pdb_path: str) -> float:
    vals = []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            b = line[60:66].strip()
            if not b:
                continue
            vals.append(float(b))
    if not vals:
        raise RuntimeError(f"No pLDDT values found in {pdb_path}")
    return float(np.mean(vals))


def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    if P.shape != Q.shape:
        raise ValueError(f"Shape mismatch in RMSD: {P.shape} vs {Q.shape}")
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    C = Pc.T @ Qc
    V, _, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])
    U = V @ D @ Wt
    P_rot = Pc @ U
    diff = P_rot - Qc
    return float(np.sqrt((diff * diff).sum() / P.shape[0]))


def simple_kmeans_1d(x, k=3, max_iter=100, n_init=20, random_state=0):
    rng = np.random.RandomState(random_state)
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    n = x.shape[0]
    k_eff = min(k, n)
    best_inertia = None
    best_labels = None
    best_centroids = None

    for _ in range(n_init):
        idx = rng.choice(n, size=k_eff, replace=False)
        centroids = x[idx].copy()

        for _ in range(max_iter):
            dists = np.linalg.norm(x[:, None, :] - centroids[None, :, :], axis=2)
            labels = np.argmin(dists, axis=1)
            new_centroids = np.zeros_like(centroids)
            for j in range(k_eff):
                mask = labels == j
                if np.any(mask):
                    new_centroids[j] = x[mask].mean(axis=0)
                else:
                    new_centroids[j] = x[rng.randint(n)]
            if np.allclose(new_centroids, centroids):
                break
            centroids = new_centroids

        inertia = ((x - centroids[labels]) ** 2).sum()
        if best_inertia is None or inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centroids = centroids.copy()

    return best_labels, best_centroids


def main():
    ap = argparse.ArgumentParser(
        description="Pick best pLDDT model, compute RMSD to best, cluster RMSD, and pick 3 reps."
    )
    ap.add_argument("--uniprot", required=True)
    ap.add_argument("--pred-dir", required=True, help="Directory with predicted PDBs.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--random-state", type=int, default=0)
    ap.add_argument("--min-plddt", type=float, default=0.0,
                    help="Drop models below this mean pLDDT before clustering. "
                         "Default 0 (keep everything) so results stay reproducible; "
                         "raise it to stop badly-folded models becoming a 'state'.")
    ap.add_argument("--warn-plddt", type=float, default=70.0,
                    help="Warn if a chosen representative falls below this mean pLDDT.")
    ap.add_argument(
        "--max-coil-fraction",
        type=float,
        default=float(os.environ.get("MAX_COIL_FRACTION", "0.60")),
        help="Discard models whose DSSP coil-like fraction exceeds this value "
             "(C/S/T/unassigned). Default 0.60 matches the paper; set 1.0 to disable.",
    )
    args = ap.parse_args()

    if not (0.0 <= args.max_coil_fraction <= 1.0):
        raise SystemExit("ERROR: --max-coil-fraction must be in [0, 1]")

    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdb_paths = sorted(glob.glob(str(pred_dir / "*.pdb")))
    if not pdb_paths:
        raise RuntimeError(f"No PDBs found in {pred_dir}")

    # Estimate chain length from the first model (CA count). Short peptides are
    # coil-dominated; applying the paper 60% filter empties the ensemble.
    n_res = int(get_ca_coords(pdb_paths[0]).shape[0])
    if args.max_coil_fraction < 1.0 and n_res < MIN_RESIDUES_FOR_COIL_FILTER:
        print(
            f"[{args.uniprot}] NOTE: sequence has {n_res} residues "
            f"(<{MIN_RESIDUES_FOR_COIL_FILTER}); DSSP coil filter is not "
            f"applicable and will be skipped. Use a longer protein to apply "
            f"the paper 0.60 cutoff.",
            flush=True,
        )
        args.max_coil_fraction = 1.0

    dssp_bin = None
    if args.max_coil_fraction < 1.0:
        dssp_bin = find_dssp_bin()
        print(f"[{args.uniprot}] DSSP filter enabled (max coil fraction "
              f"{args.max_coil_fraction:.2f}) using {dssp_bin}", flush=True)
    else:
        print(f"[{args.uniprot}] DSSP coil filter disabled (--max-coil-fraction 1.0)",
              flush=True)

    records = []
    n_dssp_fail = 0
    for p in pdb_paths:
        mean_plddt = parse_mean_plddt_from_pdb(str(p))
        coil_frac = None
        keep = True
        if dssp_bin is not None:
            try:
                coil_frac = coil_fraction(str(p), dssp_bin=dssp_bin)
                keep = coil_frac <= args.max_coil_fraction
            except Exception as e:
                n_dssp_fail += 1
                keep = False
                print(f"[{args.uniprot}] WARNING: DSSP failed on {Path(p).name} "
                      f"({e}); discarding model.", flush=True)
        records.append({
            "model": Path(p).name,
            "pdb_path": str(p),
            "mean_plddt": mean_plddt,
            "coil_fraction": coil_frac,
            "passed_coil_filter": keep,
        })

    df_all_input = pd.DataFrame(records)
    n_before = len(df_all_input)
    df = df_all_input[df_all_input["passed_coil_filter"]].copy()
    n_dropped = n_before - len(df)
    if dssp_bin is not None:
        print(f"[{args.uniprot}] DSSP coil filter <= {args.max_coil_fraction}: "
              f"kept {len(df)}/{n_before} models "
              f"(dropped {n_dropped}; DSSP failures {n_dssp_fail})", flush=True)
        # Audit trail for every input model (including rejects).
        df_all_input.sort_values("mean_plddt", ascending=False).to_csv(
            out_dir / f"{args.uniprot}_dssp_coil_filter.tsv", sep="\t", index=False
        )

    if df.empty:
        coils = df_all_input["coil_fraction"].dropna()
        detail = (
            f"No models left after DSSP coil filter "
            f"(--max-coil-fraction {args.max_coil_fraction}; "
            f"kept 0/{n_before}, DSSP failures {n_dssp_fail})."
        )
        if n_dssp_fail == 0 and len(coils) > 0:
            detail += (
                f" DSSP itself succeeded; coil fractions "
                f"min/median/max="
                f"{float(coils.min()):.2f}/"
                f"{float(coils.median()):.2f}/"
                f"{float(coils.max()):.2f}. "
                f"Typical for very short or disordered sequences where most "
                f"residues are coil-like. Set MAX_COIL_FRACTION=1.0 to disable "
                f"the filter, or use a longer folded protein."
            )
        else:
            detail += (
                " Check the DSSP install, or set MAX_COIL_FRACTION=1.0 to disable."
            )
        raise RuntimeError(detail)

    df = df.sort_values("mean_plddt", ascending=False).reset_index(drop=True)

    if args.min_plddt > 0:
        n_before = len(df)
        df = df[df["mean_plddt"] >= args.min_plddt].reset_index(drop=True)
        print(f"[{args.uniprot}] pLDDT filter >= {args.min_plddt}: kept {len(df)}/{n_before} models")
        if df.empty:
            raise RuntimeError(
                f"No models left after --min-plddt {args.min_plddt}; lower it or check the predictions."
            )

    best_row = df.iloc[0]
    best_path = best_row["pdb_path"]
    best_ca = get_ca_coords(best_path)
    print(f"[{args.uniprot}] Best reference model: {best_row['model']}")

    rmsd_vals = []
    for _, row in df.iterrows():
        coords = get_ca_coords(row["pdb_path"])
        if coords.shape != best_ca.shape:
            raise RuntimeError(f"Shape mismatch vs best for {row['pdb_path']}")
        rmsd_vals.append(kabsch_rmsd(coords, best_ca))

    df["rmsd_to_best"] = rmsd_vals
    df = df.sort_values(["mean_plddt", "rmsd_to_best"], ascending=[False, True]).reset_index(drop=True)

    df_best = df.iloc[[0]].copy()
    df_others = df.iloc[1:].copy()
    if len(df_others) < args.k:
        raise RuntimeError(f"Need at least {args.k + 1} valid models total to cluster into {args.k} states.")

    X = df_others["rmsd_to_best"].to_numpy()
    labels, centroids = simple_kmeans_1d(X, k=args.k, random_state=args.random_state)

    order = np.argsort(centroids[:, 0])
    label_map = {old: new for new, old in enumerate(order)}
    df_others["cluster"] = np.array([label_map[l] for l in labels])

    rep_rows = []
    for cl in sorted(df_others["cluster"].unique()):
        sub = df_others[df_others["cluster"] == cl].copy()
        sub = sub.sort_values(["mean_plddt", "rmsd_to_best"], ascending=[False, True])
        rep_rows.append(sub.iloc[0])

    reps_df = pd.DataFrame(rep_rows).reset_index(drop=True)
    reps_df["rep_id"] = [f"rep_cluster{i+1}" for i in range(len(reps_df))]

    # A cluster far from the reference can be a genuine alternative conformation
    # or simply a badly-folded model. Downstream ΔNC cannot tell the difference,
    # so say so loudly rather than silently treating it as a state.
    for _, r in reps_df.iterrows():
        if r["mean_plddt"] < args.warn_plddt:
            print(
                f"[{args.uniprot}] WARNING: {r['rep_id']} has mean pLDDT "
                f"{r['mean_plddt']:.1f} (< {args.warn_plddt}) at RMSD "
                f"{r['rmsd_to_best']:.1f} A from the reference. Every delta-NC value "
                f"involving it may reflect a poor prediction, not an alternative state."
            )

    df_best["cluster"] = -1
    df_best["rep_id"] = "best_ref"

    df_all = pd.concat([df_best, df_others], ignore_index=True)
    df_all = df_all.merge(reps_df[["model", "rep_id"]], on="model", how="left", suffixes=("", "_rep"))
    df_all["rep_id"] = df_all["rep_id"].fillna(df_all.get("rep_id_rep"))
    if "rep_id_rep" in df_all.columns:
        df_all = df_all.drop(columns=["rep_id_rep"])

    # Keep audit columns when present.
    out_cols = ["model", "pdb_path", "mean_plddt", "rmsd_to_best", "cluster", "rep_id"]
    if "coil_fraction" in df_all.columns:
        out_cols.insert(3, "coil_fraction")
    df_all.to_csv(out_dir / f"{args.uniprot}_plddt_rmsd_bestref.tsv", sep="\t",
                  columns=[c for c in out_cols if c in df_all.columns], index=False)

    rep_info = reps_df[["rep_id", "model", "pdb_path", "mean_plddt", "rmsd_to_best", "cluster"]].copy()
    if "coil_fraction" in reps_df.columns:
        rep_info.insert(4, "coil_fraction", reps_df["coil_fraction"])
    rep_info.to_csv(out_dir / f"{args.uniprot}_rep_info.tsv", sep="\t", index=False)

    plt.figure(figsize=(6, 4))
    mask_others = df_all["cluster"] != -1
    plt.scatter(
        df_all.loc[mask_others, "rmsd_to_best"],
        df_all.loc[mask_others, "mean_plddt"],
        c=df_all.loc[mask_others, "cluster"],
        cmap="tab10",
        s=10,
        alpha=0.7,
    )

    for _, r in reps_df.iterrows():
        plt.scatter(
            r["rmsd_to_best"],
            r["mean_plddt"],
            s=70,
            edgecolor="black",
            facecolor="none",
            linewidth=1.3,
        )
        plt.text(
            r["rmsd_to_best"],
            r["mean_plddt"] + 0.5,
            r["rep_id"],
            ha="center",
            va="bottom",
            fontsize=8,
        )

    r_best = df_best.iloc[0]
    plt.scatter(
        r_best["rmsd_to_best"],
        r_best["mean_plddt"],
        marker="*",
        s=140,
        edgecolor="black",
        facecolor="yellow",
        linewidth=1.2,
    )
    plt.text(
        r_best["rmsd_to_best"],
        r_best["mean_plddt"] + 0.5,
        "best_ref",
        ha="center",
        va="bottom",
        fontsize=8,
    )

    plt.xlabel("RMSD to best predicted (Å)")
    plt.ylabel("Mean pLDDT")
    plt.title(f"{args.uniprot} – pLDDT vs RMSD (reference excluded from clustering)")
    plt.tight_layout()
    plt.savefig(out_dir / f"{args.uniprot}_plddt_vs_rmsd_bestref.png", dpi=300)
    plt.close()

    print(f"[DONE] Wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
