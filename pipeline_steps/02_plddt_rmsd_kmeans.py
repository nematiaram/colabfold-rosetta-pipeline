#!/usr/bin/env python3
import argparse
import glob
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdb_paths = sorted(glob.glob(str(pred_dir / "*.pdb")))
    if not pdb_paths:
        raise RuntimeError(f"No PDBs found in {pred_dir}")

    records = []
    for p in pdb_paths:
        records.append({
            "model": Path(p).name,
            "pdb_path": str(p),
            "mean_plddt": parse_mean_plddt_from_pdb(str(p)),
        })

    df = pd.DataFrame(records).sort_values("mean_plddt", ascending=False).reset_index(drop=True)

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

    df_all.to_csv(out_dir / f"{args.uniprot}_plddt_rmsd_bestref.tsv", sep="\t", index=False)

    rep_info = reps_df[["rep_id", "model", "pdb_path", "mean_plddt", "rmsd_to_best", "cluster"]].copy()
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
