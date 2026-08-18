#!/usr/bin/env python3
"""
Pairwise |ΔNC| reporter ranking with reagent assignment.

For each residue, compute d12/d13/d23 = |NC_rep1 - NC_rep2| etc.
Rank residues per pair; map reporters to labeling reagents via reagent_map.py.
No absolute exposed/buried gates — discrimination is purely pairwise.

Default threshold T=5 matches the benchmark paper.
"""
import argparse
from pathlib import Path

import pandas as pd

from reagent_map import assign, normalize_aa

PAIRS = [
    ("d12", "1-2", "more_exposed_in_12"),
    ("d13", "1-3", "more_exposed_in_13"),
    ("d23", "2-3", "more_exposed_in_23"),
]


def read_nc(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if "chain" not in df.columns:
        df["chain"] = "A"
    df["resname"] = df["resname"].map(normalize_aa)
    df["resnum"] = pd.to_numeric(df["resnum"]).astype(int)
    for c in ["nc_rep1", "nc_rep2", "nc_rep3"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values(["chain", "resnum"]).reset_index(drop=True)


def residue_label(row: pd.Series) -> str:
    chain = str(row.get("chain", "")).strip()
    prefix = f"{chain}:" if chain else ""
    return f"{prefix}{row['resname']} {int(row['resnum'])}"


def more_exposed(n_a: float, n_b: float, rep_a: str, rep_b: str) -> str:
    if n_a < n_b:
        return rep_a
    if n_b < n_a:
        return rep_b
    return "tie"


def annotate(nc_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in nc_df.iterrows():
        n1, n2, n3 = float(r.nc_rep1), float(r.nc_rep2), float(r.nc_rep3)
        d12, d13, d23 = abs(n1 - n2), abs(n1 - n3), abs(n2 - n3)
        max_d = max(d12, d13, d23)
        rx = assign(r.resname)
        rows.append({
            "chain": r.get("chain", "A"),
            "resnum": int(r.resnum),
            "resname": normalize_aa(r.resname),
            "Residue": residue_label(r),
            "nc_rep1": n1, "nc_rep2": n2, "nc_rep3": n3,
            "d12": d12, "d13": d13, "d23": d23,
            "max_dNC": max_d,
            "more_exposed_in_12": more_exposed(n1, n2, "rep1", "rep2"),
            "more_exposed_in_13": more_exposed(n1, n3, "rep1", "rep3"),
            "more_exposed_in_23": more_exposed(n2, n3, "rep2", "rep3"),
            **rx,
        })
    return pd.DataFrame(rows)


def categories_above_t(row: pd.Series, threshold: float):
    cats = []
    if row["d12"] >= threshold:
        cats.append("1-2")
    if row["d13"] >= threshold:
        cats.append("1-3")
    if row["d23"] >= threshold:
        cats.append("2-3")
    return cats


def reporters_table(df: pd.DataFrame, uid: str, threshold: float) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        cats = categories_above_t(r, threshold)
        if not cats:
            continue
        pairs = [
            (r["d12"], "1-2", r["more_exposed_in_12"]),
            (r["d13"], "1-3", r["more_exposed_in_13"]),
            (r["d23"], "2-3", r["more_exposed_in_23"]),
        ]
        strongest = max(pairs, key=lambda x: x[0])
        rows.append({
            "uniprot": uid,
            "Residue": r["Residue"],
            "resname": r["resname"],
            "resnum": int(r["resnum"]),
            "nc_rep1": r["nc_rep1"],
            "nc_rep2": r["nc_rep2"],
            "nc_rep3": r["nc_rep3"],
            "d12_rep1_vs_rep2": r["d12"],
            "d13_rep1_vs_rep3": r["d13"],
            "d23_rep2_vs_rep3": r["d23"],
            "max_dNC": r["max_dNC"],
            "above_threshold": True,
            "categories_above_T": ",".join(cats),
            "clears_1-2": "1-2" in cats,
            "clears_1-3": "1-3" in cats,
            "clears_2-3": "2-3" in cats,
            "strongest_category": strongest[1],
            "strongest_dNC": strongest[0],
            "more_labeled_rep": strongest[2],
            "preferred_reagent": r["preferred_reagent"],
            "all_reagents": r["reagents"],
        })
    return pd.DataFrame(rows)


def reagent_residue_detail(reporters: pd.DataFrame, uid: str, threshold: float) -> pd.DataFrame:
    rows = []
    for _, r in reporters.iterrows():
        reagents = [x.strip() for x in str(r["all_reagents"]).split(";") if x.strip()]
        if not reagents:
            reagents = [r["preferred_reagent"]] if r["preferred_reagent"] else []
        for reagent in reagents:
            d12, d13, d23 = r["d12_rep1_vs_rep2"], r["d13_rep1_vs_rep3"], r["d23_rep2_vs_rep3"]
            pair_hits = []
            if d12 >= threshold:
                pair_hits.append(f"rep1/rep2: more-labeled={r['more_labeled_rep']} (Δ={d12:.2f})")
            if d13 >= threshold:
                pair_hits.append(f"rep1/rep3: more-labeled={r['more_labeled_rep']} (Δ={d13:.2f})")
            if d23 >= threshold:
                pair_hits.append(f"rep2/rep3: more-labeled={r['more_labeled_rep']} (Δ={d23:.2f})")
            rows.append({
                "uniprot": uid,
                "reagent": reagent,
                "Residue": r["Residue"],
                "resname": r["resname"],
                "resnum": int(r["resnum"]),
                "nc_rep1": r["nc_rep1"],
                "nc_rep2": r["nc_rep2"],
                "nc_rep3": r["nc_rep3"],
                "d12": d12, "d13": d13, "d23": d23,
                "max_d": r["max_dNC"],
                "n_pairs_above_threshold": len(pair_hits),
                "pairs_above_threshold": "; ".join(pair_hits),
                "preferred_reagent": r["preferred_reagent"],
                "threshold": threshold,
            })
    return pd.DataFrame(rows)


def reagent_target_counts(detail: pd.DataFrame, uid: str) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame(columns=[
            "uniprot", "reagent", "Total_targetable_unique_residues",
            "Rep_1-2", "Rep_2-3", "Rep_1-3", "Pairs_covered", "Tier",
        ])
    from reagent_map import NONSPEC_ORDER, REAGENT_ORDER

    rows = []
    for reagent, sub in detail.groupby("reagent"):
        residues = sub.drop_duplicates(["resname", "resnum"])
        n12 = int((residues["d12"] >= sub["threshold"].iloc[0]).sum())
        n23 = int((residues["d23"] >= sub["threshold"].iloc[0]).sum())
        n13 = int((residues["d13"] >= sub["threshold"].iloc[0]).sum())
        pairs = sum(x > 0 for x in (n12, n23, n13))
        tier = "non-specific" if reagent in NONSPEC_ORDER else "specific"
        rows.append({
            "uniprot": uid,
            "reagent": reagent,
            "Total_targetable_unique_residues": len(residues),
            "Rep_1-2": n12,
            "Rep_2-3": n23,
            "Rep_1-3": n13,
            "Pairs_covered": pairs,
            "Tier": tier,
        })
    out = pd.DataFrame(rows)
    order = {r: i for i, r in enumerate(REAGENT_ORDER + NONSPEC_ORDER)}
    out["_ord"] = out["reagent"].map(lambda r: order.get(r, 999))
    return out.sort_values(["_ord", "Total_targetable_unique_residues"],
                           ascending=[True, False]).drop(columns=["_ord"])


def pick_best_per_pair(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    picks = []
    pair_specs = [
        ("rep1_vs_rep2", "d12", "more_exposed_in_12", "rep1/rep2"),
        ("rep1_vs_rep3", "d13", "more_exposed_in_13", "rep1/rep3"),
        ("rep2_vs_rep3", "d23", "more_exposed_in_23", "rep2/rep3"),
    ]
    for pair_name, dcol, exp_col, pair_short in pair_specs:
        ranked = df[df[dcol] >= threshold].copy()
        if ranked.empty:
            continue
        ranked["_spec"] = ranked["has_specific"].astype(int)
        ranked = ranked.sort_values([dcol, "_spec", "resnum"],
                                    ascending=[False, False, True])
        choice = ranked.iloc[0]
        picks.append({
            "pair": pair_name,
            "pair_short": pair_short,
            "Residue": choice["Residue"],
            "resname": choice["resname"],
            "resnum": int(choice["resnum"]),
            "pair_diff": float(choice[dcol]),
            "more_exposed_rep": choice[exp_col],
            "preferred_reagent": choice["preferred_reagent"],
            "reagent_tier": choice["reagent_tier"],
            "all_reagents": choice["reagents"],
        })
    return pd.DataFrame(picks)


def write_recommendation(uid: str, picks: pd.DataFrame, path: Path) -> None:
    lines = [f"# {uid} pairwise NC recommendations (|ΔNC| ≥ threshold)", ""]
    if picks.empty:
        lines.append("No residues cleared threshold for any pair.")
    else:
        for _, p in picks.iterrows():
            lines.append(
                f"{p['pair_short']}: {p['Residue']}  ΔNC={p['pair_diff']:.2f}  "
                f"more labeled in {p['more_exposed_rep']}  →  {p['preferred_reagent']}"
            )
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Pairwise ΔNC ranking + reagent mapping.")
    ap.add_argument("--uniprot", required=True)
    ap.add_argument("--nc-tsv", required=True, help="Merged Rosetta NC table from step 03.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--threshold", type=float, default=5.0,
                    help="Minimum |ΔNC| for a reporter (default: 5).")
    args = ap.parse_args()

    uid = args.uniprot
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = annotate(read_nc(Path(args.nc_tsv)))
    df.to_csv(out_dir / f"{uid}_all_residues_with_pair_diffs.tsv", sep="\t", index=False)

    reporters = reporters_table(df, uid, args.threshold)
    reporters.to_csv(out_dir / f"{uid}_residues_dNC_ge_{int(args.threshold)}.tsv",
                     sep="\t", index=False)
    reporters.to_csv(out_dir / f"{uid}_above_threshold_residues.tsv", sep="\t", index=False)

    detail = reagent_residue_detail(reporters, uid, args.threshold)
    detail.to_csv(out_dir / f"{uid}_reagent_residue_detail.tsv", sep="\t", index=False)

    counts = reagent_target_counts(detail, uid)
    counts.to_csv(out_dir / f"{uid}_reagent_target_counts.tsv", sep="\t", index=False)

    picks = pick_best_per_pair(df, args.threshold)
    picks.to_csv(out_dir / f"{uid}_best_per_pair.tsv", sep="\t", index=False)
    write_recommendation(uid, picks, out_dir / f"{uid}_per_pair_recommendation.txt")

    for pair_name, dcol, _ in [
        ("rep1_vs_rep2", "d12", "1-2"),
        ("rep1_vs_rep3", "d13", "1-3"),
        ("rep2_vs_rep3", "d23", "2-3"),
    ]:
        ranked = df.sort_values(dcol, ascending=False).reset_index(drop=True)
        ranked.insert(0, "rank", range(1, len(ranked) + 1))
        ranked.to_csv(out_dir / f"{uid}_rank_{pair_name}.tsv", sep="\t", index=False)

    print(f"[DONE] {uid}: {len(reporters)} reporters at T={args.threshold}")
    print(f"  wrote {out_dir}")


if __name__ == "__main__":
    main()
