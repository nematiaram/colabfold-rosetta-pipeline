#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.colors import Normalize, to_hex
from matplotlib import cm


REPS = ["rep1", "rep2", "rep3"]
AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

REAGENT_ORDER = [
    "DEPC",
    "N-acetylimidazole",
    "Phenylglyoxal",
    "p-hydroxyphenylglyoxal",
    "2,3-butanedione",
    "1,2-cyclohexanedione",
    "Methylglyoxal",
    "Kethoxal",
    "Iodoacetamide/iodoacetate",
    "Acryloyl",
    "Acetic anhydride",
    "Succinic anhydride",
    "Maleic anhydride",
    "S-methylthiocacetimidate",
    "N-bromosuccinimide (NBS)",
    "EDC (carbodiimide)",
    "Koshland's reagent (HNB bromide)",
    "O-nitrophenylsulfenyl chloride",
    "Tetranitromethane",
    "Iodine",
]
REAGENT_TO_STAR = {name: f"*{i+1}" for i, name in enumerate(REAGENT_ORDER)}
STAR_TO_REAGENT = {v: k for k, v in REAGENT_TO_STAR.items()}

NON_SPECIFIC_PRIORITY = ["OH-high", "OH-medium", "OH-low", "diazirine", "CF3"]

COLORS = {
    "bg": "#FBFBFD",
    "panel": "#FFFFFF",
    "line": "#495057",
    "title": "#243447",
    "text": "#1F2937",
    "muted": "#6B7280",
    "labeled": "#1F9D55",
    "label_fill": "#E7F7EF",
    "not_labeled": "#C0392B",
    "nolabel_fill": "#FCEBE8",
    "step_fill": "#EEF3FB",
    "decision_fill": "#FFF7E6",
    "header": "#2E5E8E",
    "row_alt": "#F7EFCB",
    "grid": "#000000",
}

# Built-in reagent knowledge from the user's tables/screenshots.
# These are used only when --labels-source is not provided.
BUILTIN_SPECIFIC = {
    "ARG": [
        "Phenylglyoxal",
        "p-hydroxyphenylglyoxal",
        "2,3-butanedione",
        "1,2-cyclohexanedione",
        "Methylglyoxal",
        "Kethoxal",
        "DEPC",
        "N-bromosuccinimide (NBS)",
    ],
    "ASP": ["EDC (carbodiimide)"],
    "GLU": ["EDC (carbodiimide)"],
    "CYS": [
        "EDC (carbodiimide)",
        "Iodoacetamide/iodoacetate",
        "Acryloyl",
        "Acetic anhydride",
        "Succinic anhydride",
        "Maleic anhydride",
        "Koshland's reagent (HNB bromide)",
        "N-bromosuccinimide (NBS)",
        "O-nitrophenylsulfenyl chloride",
        "Iodine",
    ],
    "HIS": [
        "DEPC",
        "2,3-butanedione",
        "Iodoacetamide/iodoacetate",
        "N-bromosuccinimide (NBS)",
        "Tetranitromethane",
        "Iodine",
    ],
    "LYS": [
        "Phenylglyoxal",
        "2,3-butanedione",
        "1,2-cyclohexanedione",
        "Methylglyoxal",
        "Iodoacetamide/iodoacetate",
        "Acryloyl",
        "DEPC",
        "Acetic anhydride",
        "Succinic anhydride",
        "Maleic anhydride",
        "S-methylthiocacetimidate",
        "N-bromosuccinimide (NBS)",
        "N-acetylimidazole",
    ],
    "SER": ["DEPC", "N-acetylimidazole"],
    "THR": ["DEPC", "N-acetylimidazole"],
    "TRP": [
        "Koshland's reagent (HNB bromide)",
        "N-bromosuccinimide (NBS)",
        "O-nitrophenylsulfenyl chloride",
        "Tetranitromethane",
        "Iodine",
    ],
    "TYR": [
        "EDC (carbodiimide)",
        "Iodoacetamide/iodoacetate",
        "DEPC",
        "Acetic anhydride",
        "Succinic anhydride",
        "Maleic anhydride",
        "Koshland's reagent (HNB bromide)",
        "N-bromosuccinimide (NBS)",
        "Tetranitromethane",
        "Iodine",
        "N-acetylimidazole",
    ],
    "MET": ["Iodoacetamide/iodoacetate", "Tetranitromethane", "Iodine"],
}

BUILTIN_NONSPEC = {
    "CYS": ["OH-high", "diazirine"],
    "TRP": ["OH-high", "diazirine"],
    "TYR": ["OH-high", "diazirine"],
    "MET": ["OH-high", "diazirine"],
    "PHE": ["OH-high", "diazirine"],
    "HIS": ["OH-high", "diazirine"],
    "ARG": ["OH-high", "diazirine"],
    "ILE": ["OH-high", "diazirine"],
    "LEU": ["OH-high", "diazirine"],
    "VAL": ["OH-medium", "diazirine"],
    "PRO": ["OH-medium", "diazirine"],
    "GLN": ["OH-medium", "diazirine"],
    "THR": ["OH-medium", "diazirine", "CF3"],
    "LYS": ["OH-medium", "diazirine"],
    "SER": ["OH-medium", "diazirine", "CF3"],
    "GLU": ["OH-medium", "diazirine", "CF3"],
    "ALA": ["OH-low", "diazirine", "CF3"],
    "ASP": ["OH-low", "diazirine", "CF3"],
    "ASN": ["OH-low", "diazirine"],
    "GLY": ["OH-low", "diazirine", "CF3"],
    # broad diazirine coverage:
    "ALL": ["diazirine"],
}


def normalize_resname(x: object) -> str:
    s = str(x).strip().upper()
    if len(s) == 1:
        for k, v in AA3_TO_1.items():
            if v == s:
                return k
    return s


def aa_display(resname: str) -> str:
    r = normalize_resname(resname)
    if len(r) == 3 and r in AA3_TO_1:
        return f"{r} {AA3_TO_1[r]}"
    return r


def residue_label(row: pd.Series) -> str:
    chain = str(row.get("chain", "")).strip()
    prefix = f"{chain}:" if chain else ""
    return f"{prefix}{normalize_resname(row['resname'])} {int(row['resnum'])}"


def parse_star_numbers(labels_str: object) -> List[int]:
    return sorted(set(int(m.group(1)) for m in re.finditer(r"\*(\d+)", str(labels_str))))


def star_to_reagent_name(n: int) -> str:
    if 1 <= n <= len(REAGENT_ORDER):
        return f"*{n} {REAGENT_ORDER[n - 1]}"
    return f"*{n}"


def expand_reagents(labels_str: object, label_non_specific: object) -> List[str]:
    reagents: List[str] = []
    for n in parse_star_numbers(labels_str):
        reagents.append(star_to_reagent_name(n))
    parts = [p.strip() for p in str(label_non_specific).split(";") if p.strip()]
    parts_lower = [p.lower() for p in parts]
    for key in NON_SPECIFIC_PRIORITY:
        for part, part_lower in zip(parts, parts_lower):
            if part_lower == key.lower() and part not in reagents:
                reagents.append(part)
    for part in parts:
        if part not in reagents:
            reagents.append(part)
    return reagents


def parse_rep_set(s: object) -> Set[str]:
    text = str(s).strip()
    if not text or text.lower() in {"none", "nan"}:
        return set()
    return {x.strip() for x in text.split("/") if x.strip()}


def fmt_rep_set(xs: Sequence[str]) -> str:
    return " or ".join(xs) if xs else "ambiguous"


def built_in_label_entry(resname: str) -> Dict[str, str]:
    res = normalize_resname(resname)
    specific = BUILTIN_SPECIFIC.get(res, [])
    nonspec = list(BUILTIN_NONSPEC.get(res, []))
    if "diazirine" not in nonspec and res in AA3_TO_1:
        nonspec.append("diazirine")
    labels = "; ".join(REAGENT_TO_STAR[x] for x in specific if x in REAGENT_TO_STAR)
    label_non_specific = "; ".join([x for x in NON_SPECIFIC_PRIORITY if x in nonspec] +
                                   [x for x in nonspec if x not in NON_SPECIFIC_PRIORITY])
    return {"labels": labels, "label_non_specific": label_non_specific}


def read_nc_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    required = {"resname", "resnum", "nc_rep1", "nc_rep2", "nc_rep3"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required NC columns: {sorted(missing)}")
    if "chain" not in df.columns:
        df["chain"] = ""
    df["resname"] = df["resname"].apply(normalize_resname)
    df["resnum"] = pd.to_numeric(df["resnum"], errors="raise").astype(int)
    for c in ["nc_rep1", "nc_rep2", "nc_rep3"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values(["chain", "resnum"]).reset_index(drop=True)


def read_reagent_map(path: Optional[Path]) -> Dict[str, Dict[str, str]]:
    if path is None:
        return {aa: built_in_label_entry(aa) for aa in AA3_TO_1}
    df = pd.read_csv(path, sep=None, engine="python")
    required = {"resname", "labels", "label_non_specific"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Reagent map is missing columns: {sorted(missing)}")
    out: Dict[str, Dict[str, str]] = {}
    for _, row in df.iterrows():
        out[normalize_resname(row["resname"])] = {
            "labels": "" if pd.isna(row["labels"]) else str(row["labels"]),
            "label_non_specific": "" if pd.isna(row["label_non_specific"]) else str(row["label_non_specific"]),
        }
    return out


def compute_rank_table(nc_df: pd.DataFrame, tie_eps: float = 1e-9) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for _, row in nc_df.iterrows():
        vals = {"rep1": float(row["nc_rep1"]), "rep2": float(row["nc_rep2"]), "rep3": float(row["nc_rep3"])}
        ordered = sorted(vals.items(), key=lambda kv: kv[1])
        min_rep, min_val = ordered[0]
        mid_rep, mid_val = ordered[1]
        max_rep, max_val = ordered[2]
        low_diff = mid_val - min_val
        high_diff = max_val - mid_val
        margin = abs(high_diff - low_diff)
        if abs(high_diff - low_diff) <= tie_eps:
            winner_case = "ambiguous"
            winner_rep = ""
            winner_diff = max(low_diff, high_diff)
            interpretation = "ambiguous"
        elif high_diff > low_diff:
            winner_case = "max"
            winner_rep = max_rep
            winner_diff = high_diff
            interpretation = f"buried in {max_rep}"
        else:
            winner_case = "min"
            winner_rep = min_rep
            winner_diff = low_diff
            interpretation = f"exposed in {min_rep}"
        rows.append({
            "chain": row.get("chain", ""),
            "resnum": int(row["resnum"]),
            "resname": row["resname"],
            "Residue": residue_label(row),
            "rep1": vals["rep1"],
            "rep2": vals["rep2"],
            "rep3": vals["rep3"],
            "min_rep": min_rep,
            "middle_rep": mid_rep,
            "max_rep": max_rep,
            "min_val": min_val,
            "middle_val": mid_val,
            "max_val": max_val,
            "low_diff": low_diff,
            "high_diff": high_diff,
            "winner_case": winner_case,
            "winner_rep": winner_rep,
            "winner_diff": winner_diff,
            "margin": margin,
            "interpretation": interpretation,
        })
    return pd.DataFrame(rows).sort_values(["winner_diff", "margin", "resnum"],
                                          ascending=[False, False, True]).reset_index(drop=True)


def attach_reagent_columns(rank_df: pd.DataFrame, reagent_map: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    labels, nonspec, reagent_list = [], [], []
    for _, row in rank_df.iterrows():
        hit = reagent_map.get(normalize_resname(row["resname"]), built_in_label_entry(row["resname"]))
        labels_val = hit.get("labels", "")
        nonspec_val = hit.get("label_non_specific", "")
        labels.append(labels_val)
        nonspec.append(nonspec_val)
        reagent_list.append("; ".join(expand_reagents(labels_val, nonspec_val)))
    out = rank_df.copy()
    out["labels"] = labels
    out["label_non_specific"] = nonspec
    out["reagents"] = reagent_list
    return out


def build_top_tables(uid: str, rank_df: pd.DataFrame, out_dir: Path, top_n: int,
                     min_winner_diff: float = 0.0, min_margin: float = 0.0) -> pd.DataFrame:
    filtered = rank_df[(rank_df["winner_case"] != "ambiguous") &
                       (rank_df["winner_diff"] >= min_winner_diff) &
                       (rank_df["margin"] >= min_margin)].copy()
    top_frames = []
    for rep in REPS:
        sub = filtered[filtered["winner_rep"] == rep].copy()
        sub = sub.sort_values(["winner_diff", "margin", "resnum"], ascending=[False, False, True]).head(top_n)
        sub.insert(0, "target_rep", rep)
        sub.insert(1, "rank_in_rep", range(1, len(sub) + 1))
        sub["case"] = sub["winner_case"]
        sub["diff"] = sub["winner_diff"]
        sub["rep"] = sub["winner_rep"]
        sub.to_csv(out_dir / f"{uid}_{rep}_top{top_n}.tsv", sep="\t", index=False)
        top_frames.append(sub)
    if top_frames:
        top_df = pd.concat(top_frames, ignore_index=True)
    else:
        top_df = pd.DataFrame(columns=["target_rep", "rank_in_rep"] + list(rank_df.columns))
    top_df.to_csv(out_dir / f"{uid}_top{top_n}_all_reps.tsv", sep="\t", index=False)
    return top_df



def safe_name(s: str) -> str:
    s = s.strip().replace("/", "_")
    s = re.sub(r"[^A-Za-z0-9_.+-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "reagent"


def build_reagent_residue_tables(uid: str, rank_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """
    For each reagent bucket (specific or non-specific), collect all residues assigned to it
    and report which representative the residue is informative for.
    """
    rows = []
    informative = rank_df[rank_df["winner_case"] != "ambiguous"].copy()
    for _, r in informative.iterrows():
        residue = residue_label(r)
        state = "buried" if str(r["winner_case"]) == "max" else "exposed"
        reagents = expand_reagents(r.get("labels", ""), r.get("label_non_specific", ""))
        for reagent in reagents:
            rows.append({
                "reagent": reagent,
                "residue": residue,
                "resname": r["resname"],
                "aa1": aa_display(r["resname"]),
                "rep": r["winner_rep"],
                "state": state,
                "diff": round(float(r["winner_diff"]), 3),
                "margin": round(float(r["margin"]), 3),
                "case": r["winner_case"],
                "rep1": round(float(r["rep1"]), 3),
                "rep2": round(float(r["rep2"]), 3),
                "rep3": round(float(r["rep3"]), 3),
            })

    reagent_df = pd.DataFrame(rows)
    if reagent_df.empty:
        reagent_df = pd.DataFrame(columns=[
            "reagent", "residue", "resname", "aa1", "rep", "state", "diff", "margin",
            "case", "rep1", "rep2", "rep3"
        ])
    else:
        reagent_df = reagent_df.sort_values(
            ["reagent", "rep", "diff", "margin", "residue"],
            ascending=[True, True, False, False, True]
        ).reset_index(drop=True)

    all_path = out_dir / f"{uid}_reagent_residue_table_all.tsv"
    reagent_df.to_csv(all_path, sep="	", index=False)

    for reagent, sub in reagent_df.groupby("reagent", sort=True):
        out = out_dir / f"{uid}_reagent_{safe_name(reagent)}_residue_table.tsv"
        sub.to_csv(out, sep="	", index=False)

    return reagent_df

def build_reagent_rules_table(top_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in top_df.iterrows():
        reagents = expand_reagents(row.get("labels", ""), row.get("label_non_specific", ""))
        if not reagents:
            continue
        winner_case = str(row["winner_case"])
        winner_rep = str(row["winner_rep"])
        if winner_case == "min":
            if_label = winner_rep
            if_no_label = "/".join(sorted(set(REPS) - {winner_rep}))
        elif winner_case == "max":
            if_label = "/".join(sorted(set(REPS) - {winner_rep}))
            if_no_label = winner_rep
        else:
            continue
        for reagent in reagents:
            rows.append({
                "reagent": reagent,
                "residue": row["Residue"],
                "resname": row["resname"],
                "resnum": int(row["resnum"]),
                "target_rep": row["winner_rep"],
                "winner_rep": row["winner_rep"],
                "winner_case": row["winner_case"],
                "interpretation": row["interpretation"],
                "if_label": if_label,
                "if_no_label": if_no_label,
                "diff": float(row["winner_diff"]),
                "margin": float(row["margin"]),
                "rep1": float(row["rep1"]),
                "rep2": float(row["rep2"]),
                "rep3": float(row["rep3"]),
                "labels": row.get("labels", ""),
                "label_non_specific": row.get("label_non_specific", ""),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return (out.sort_values(["reagent", "residue", "diff", "margin"], ascending=[True, True, False, False])
              .drop_duplicates(subset=["reagent", "residue"], keep="first")
              .reset_index(drop=True))


def best_test(rows: pd.DataFrame, candidates: Set[str]):
    best = None
    best_key = None
    for r in rows.itertuples(index=False):
        label_branch = parse_rep_set(r.if_label) & candidates
        no_label_branch = parse_rep_set(r.if_no_label) & candidates
        if not label_branch and not no_label_branch:
            continue
        worst = max(len(label_branch) if label_branch else len(candidates),
                    len(no_label_branch) if no_label_branch else len(candidates))
        best_branch = min(len(label_branch) if label_branch else len(candidates),
                          len(no_label_branch) if no_label_branch else len(candidates))
        key = (worst, -best_branch, -float(r.diff), -float(getattr(r, "margin", 0.0)))
        if best_key is None or key < best_key:
            best_key, best = key, r
    return best


def build_tree_for_reagent(subdf: pd.DataFrame, max_steps: int = 2):
    rows = subdf.copy()
    candidates = set(REPS)
    step1 = best_test(rows, candidates)
    if step1 is None:
        return None
    tree = {
        "step1_residue": step1.residue,
        "step1_if_label": sorted(list(parse_rep_set(step1.if_label) & candidates)),
        "step1_if_no_label": sorted(list(parse_rep_set(step1.if_no_label) & candidates)),
        "branches": [],
    }
    if max_steps == 1:
        return tree
    for outcome, reps_left in [
        ("label", parse_rep_set(step1.if_label) & candidates),
        ("no_label", parse_rep_set(step1.if_no_label) & candidates),
    ]:
        if len(reps_left) <= 1:
            tree["branches"].append({
                "branch": outcome, "decision": sorted(list(reps_left)),
                "step2_residue": "", "step2_if_label": [], "step2_if_no_label": [],
            })
            continue
        rows2 = rows[rows["residue"] != step1.residue]
        step2 = best_test(rows2, reps_left)
        if step2 is None:
            tree["branches"].append({
                "branch": outcome, "decision": sorted(list(reps_left)),
                "step2_residue": "", "step2_if_label": [], "step2_if_no_label": [],
            })
            continue
        tree["branches"].append({
            "branch": outcome,
            "decision": [],
            "step2_residue": step2.residue,
            "step2_if_label": sorted(list(parse_rep_set(step2.if_label) & reps_left)),
            "step2_if_no_label": sorted(list(parse_rep_set(step2.if_no_label) & reps_left)),
        })
    return tree


def tree_to_text(reagent: str, tree: dict) -> str:
    lines = [reagent]
    lines.append(f"  Step 1: test {tree['step1_residue']}")
    lines.append(f"    If LABELED     -> {fmt_rep_set(tree['step1_if_label'])}")
    lines.append(f"    If NOT LABELED -> {fmt_rep_set(tree['step1_if_no_label'])}")
    for branch in tree["branches"]:
        if branch["step2_residue"]:
            lines.append(f"  Step 2 ({branch['branch']} branch): test {branch['step2_residue']}")
            lines.append(f"    If LABELED     -> {fmt_rep_set(branch['step2_if_label'])}")
            lines.append(f"    If NOT LABELED -> {fmt_rep_set(branch['step2_if_no_label'])}")
        else:
            lines.append(f"  ({branch['branch']} branch) Decision -> {fmt_rep_set(branch['decision'])}")
    return "\n".join(lines)


def design_tests_from_rules(uid: str, rules_df: pd.DataFrame, out_dir: Path,
                            max_steps: int = 2, min_rows: int = 2) -> Tuple[Path, Path]:
    out_txt = out_dir / f"{uid}_tests_per_reagent.txt"
    out_tsv = out_dir / f"{uid}_tests_per_reagent.tsv"
    if rules_df.empty:
        out_txt.write_text("")
        pd.DataFrame().to_csv(out_tsv, sep="\t", index=False)
        return out_txt, out_tsv
    blocks, rows_out = [], []
    for reagent, sub in rules_df.groupby("reagent", sort=True):
        if len(sub) < min_rows:
            continue
        tree = build_tree_for_reagent(sub, max_steps=max_steps)
        if tree is None:
            continue
        blocks.append(tree_to_text(reagent, tree))
        row = {
            "reagent": reagent,
            "step1_residue": tree["step1_residue"],
            "step1_if_label": "/".join(tree["step1_if_label"]),
            "step1_if_no_label": "/".join(tree["step1_if_no_label"]),
        }
        for b in tree["branches"]:
            prefix = f"step2_{b['branch']}"
            row[f"{prefix}_residue"] = b["step2_residue"]
            row[f"{prefix}_if_label"] = "/".join(b["step2_if_label"])
            row[f"{prefix}_if_no_label"] = "/".join(b["step2_if_no_label"])
            row[f"{prefix}_decision"] = "/".join(b["decision"])
        rows_out.append(row)
    out_txt.write_text("\n\n".join(blocks))
    pd.DataFrame(rows_out).to_csv(out_tsv, sep="\t", index=False)
    return out_txt, out_tsv


def pretty_specific_labels(label_str: str) -> str:
    nums = parse_star_numbers(label_str)
    if not nums:
        return ""
    return "; ".join(f"*{n}" for n in nums)


def reagent_legend_lines() -> List[str]:
    return [f"*{i+1} = {name}" for i, name in enumerate(REAGENT_ORDER)]


def render_top_table_png(top_df: pd.DataFrame, out_png: Path, uid: str, top_n: int):
    cols = ["Residue", "rep", "case", "diff", "rep1", "rep2", "rep3", "labels", "label_non_specific"]
    if top_df.empty:
        fig, ax = plt.subplots(figsize=(8, 2.4))
        ax.axis("off")
        ax.text(0.5, 0.5, f"{uid}: no informative residues passed filters", ha="center", va="center", fontsize=12)
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    df = top_df.copy().sort_values(["target_rep", "rank_in_rep"]).reset_index(drop=True)
    df["rep"] = df["winner_rep"]
    df["case"] = df["winner_case"]
    df["diff"] = df["winner_diff"]
    df["labels"] = df["labels"].apply(pretty_specific_labels)
    df["label_non_specific"] = df["label_non_specific"].astype(str)
    df = df[cols]

    n_rows, n_cols = df.shape
    fig_h = max(6, 0.44 * (n_rows + 1.0) + 2.3)
    fig, ax = plt.subplots(figsize=(18, fig_h))
    ax.axis("off")

    display_df = df.copy()
    for c in display_df.columns:
        if np.issubdtype(display_df[c].dtype, np.number):
            display_df[c] = display_df[c].round(3)
        display_df[c] = display_df[c].astype(str)

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.55)

    for j in range(n_cols):
        cell = table[(0, j)]
        cell.set_facecolor(COLORS["header"])
        cell.get_text().set_color("white")
        cell.get_text().set_weight("bold")
        cell.set_edgecolor(COLORS["grid"])
        cell.set_linewidth(1.2)

    diff_idx = display_df.columns.get_loc("diff")
    diffs = pd.to_numeric(df["diff"], errors="coerce")
    vmin = float(np.nanmin(diffs)) if np.isfinite(np.nanmin(diffs)) else 0.0
    vmax = float(np.nanmax(diffs)) if np.isfinite(np.nanmax(diffs)) else 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap("RdYlGn_r")

    for i in range(1, n_rows + 1):
        row_color = COLORS["row_alt"] if i % 2 == 0 else "white"
        for j in range(n_cols):
            cell = table[(i, j)]
            cell.set_edgecolor(COLORS["grid"])
            cell.set_linewidth(1.0)
            if j == diff_idx:
                val = float(df.iloc[i - 1]["diff"])
                cell.set_facecolor(cmap(norm(val)))
                txt_color = "white" if norm(val) > 0.72 else "black"
                cell.get_text().set_color(txt_color)
                cell.get_text().set_weight("bold")
            else:
                cell.set_facecolor(row_color)
                cell.get_text().set_color("black")

    legend_lines = ["Specific labels are shown as *n in the table.", ""]
    legend_lines += reagent_legend_lines()
    legend_lines += ["", "Non-specific buckets: " + ", ".join(NON_SPECIFIC_PRIORITY)]
    fig.suptitle(f"{uid} – top {top_n} per rep", fontsize=16, weight="bold", y=0.98)
    fig.text(0.01, 0.02, "\n".join(legend_lines), fontsize=8.5, ha="left", va="bottom")
    fig.subplots_adjust(top=0.94, bottom=0.25, left=0.01, right=0.99)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_tests_txt(txt_path: Path) -> List[dict]:
    text = txt_path.read_text().strip()
    if not text:
        return []
    blocks = []
    for chunk in text.split("\n\n"):
        lines = [ln.rstrip() for ln in chunk.splitlines() if ln.strip()]
        if not lines:
            continue
        reagent = lines[0].strip()
        block = {"reagent": reagent, "step1_res": "", "step1_lab": "", "step1_nlab": "",
                 "step2_label_res": "", "step2_label_lab": "", "step2_label_nlab": "",
                 "step2_no_label_res": "", "step2_no_label_lab": "", "step2_no_label_nlab": "",
                 "label_branch": "", "no_label_branch": ""}
        current = None
        for ln in lines[1:]:
            s = ln.strip()
            if s.startswith("Step 1: test "):
                block["step1_res"] = s.replace("Step 1: test ", "", 1); current = "step1"
            elif s.startswith("If LABELED"):
                rhs = s.split("->", 1)[1].strip() if "->" in s else ""
                if current == "step1": block["step1_lab"] = rhs
                elif current == "step2_label": block["step2_label_lab"] = rhs
                elif current == "step2_no_label": block["step2_no_label_lab"] = rhs
            elif s.startswith("If NOT LABELED"):
                rhs = s.split("->", 1)[1].strip() if "->" in s else ""
                if current == "step1": block["step1_nlab"] = rhs
                elif current == "step2_label": block["step2_label_nlab"] = rhs
                elif current == "step2_no_label": block["step2_no_label_nlab"] = rhs
            elif s.startswith("Step 2 (label branch): test "):
                block["step2_label_res"] = s.replace("Step 2 (label branch): test ", "", 1); current = "step2_label"
            elif s.startswith("Step 2 (no_label branch): test "):
                block["step2_no_label_res"] = s.replace("Step 2 (no_label branch): test ", "", 1); current = "step2_no_label"
            elif s.startswith("(label branch) Decision -> "):
                block["label_branch"] = s.replace("(label branch) Decision -> ", "", 1)
            elif s.startswith("(no_label branch) Decision -> "):
                block["no_label_branch"] = s.replace("(no_label branch) Decision -> ", "", 1)
        blocks.append(block)
    return blocks


def draw_round_box(ax, center, w, h, text, fc, ec, fontsize=10, weight="normal"):
    x = center[0] - w / 2
    y = center[1] - h / 2
    patch = patches.FancyBboxPatch((x, y), w, h,
                                   boxstyle="round,pad=0.012,rounding_size=0.02",
                                   linewidth=1.5, edgecolor=ec, facecolor=fc)
    ax.add_patch(patch)
    ax.text(center[0], center[1], text, ha="center", va="center",
            fontsize=fontsize, color=COLORS["text"], fontweight=weight, wrap=True)


def draw_connector(ax, start, end, color, lw=2.0):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="-", color=color, lw=lw, shrinkA=0, shrinkB=0))


def draw_branch_label(ax, center, text, ec, fc):
    draw_round_box(ax, center, 0.15, 0.055, text, fc=fc, ec=ec, fontsize=9, weight="bold")


def render_tree(ax, block: dict):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); ax.set_facecolor(COLORS["bg"])
    ax.text(0.5, 0.95, block["reagent"], ha="center", va="top", fontsize=13, fontweight="bold", color=COLORS["title"])
    step1 = (0.5, 0.78); left_decision = (0.23, 0.15); middle_decision = (0.50, 0.15)
    right_decision = (0.77, 0.15); step2_left = (0.30, 0.42); step2_right = (0.70, 0.42)
    draw_round_box(ax, step1, 0.28, 0.09, f"Step 1\n{block['step1_res']}", COLORS["step_fill"], COLORS["line"], fontsize=10, weight="bold")
    draw_connector(ax, (0.44, 0.735), (0.30, 0.50), COLORS["labeled"])
    draw_connector(ax, (0.56, 0.735), (0.70, 0.50), COLORS["not_labeled"])
    draw_branch_label(ax, (0.29, 0.60), "LABELED", COLORS["labeled"], COLORS["label_fill"])
    draw_branch_label(ax, (0.71, 0.60), "NOT LABELED", COLORS["not_labeled"], COLORS["nolabel_fill"])
    if block.get("step2_label_res"):
        draw_round_box(ax, step2_left, 0.26, 0.09, f"Step 2\n{block['step2_label_res']}", COLORS["step_fill"], COLORS["line"], fontsize=10, weight="bold")
        draw_connector(ax, (0.26, 0.375), (0.20, 0.22), COLORS["labeled"])
        draw_connector(ax, (0.34, 0.375), (0.43, 0.22), COLORS["not_labeled"])
        draw_branch_label(ax, (0.18, 0.29), "LABELED", COLORS["labeled"], COLORS["label_fill"])
        draw_branch_label(ax, (0.42, 0.29), "NOT LABELED", COLORS["not_labeled"], COLORS["nolabel_fill"])
        draw_round_box(ax, left_decision, 0.22, 0.08, block.get("step2_label_lab", ""), COLORS["decision_fill"], COLORS["line"], fontsize=9)
        draw_round_box(ax, middle_decision, 0.22, 0.08, block.get("step2_label_nlab", ""), COLORS["decision_fill"], COLORS["line"], fontsize=9)
    else:
        text = block.get("label_branch") or block.get("step1_lab", "")
        draw_round_box(ax, step2_left, 0.28, 0.08, text, COLORS["decision_fill"], COLORS["line"], fontsize=9)
    if block.get("step2_no_label_res"):
        draw_round_box(ax, step2_right, 0.26, 0.09, f"Step 2\n{block['step2_no_label_res']}", COLORS["step_fill"], COLORS["line"], fontsize=10, weight="bold")
        draw_connector(ax, (0.66, 0.375), (0.58, 0.22), COLORS["labeled"])
        draw_connector(ax, (0.74, 0.375), (0.80, 0.22), COLORS["not_labeled"])
        draw_branch_label(ax, (0.58, 0.29), "LABELED", COLORS["labeled"], COLORS["label_fill"])
        draw_branch_label(ax, (0.82, 0.29), "NOT LABELED", COLORS["not_labeled"], COLORS["nolabel_fill"])
        draw_round_box(ax, middle_decision, 0.22, 0.08, block.get("step2_no_label_lab", ""), COLORS["decision_fill"], COLORS["line"], fontsize=9)
        draw_round_box(ax, right_decision, 0.22, 0.08, block.get("step2_no_label_nlab", ""), COLORS["decision_fill"], COLORS["line"], fontsize=9)
    else:
        text = block.get("no_label_branch") or block.get("step1_nlab", "")
        draw_round_box(ax, step2_right, 0.28, 0.08, text, COLORS["decision_fill"], COLORS["line"], fontsize=9)


def render_tests_per_reagent_tree_pretty(txt_path: Path, per_page: int = 4) -> List[Path]:
    blocks = parse_tests_txt(txt_path)
    if not blocks:
        return []
    uid = txt_path.name.split("_")[0]
    out_png = txt_path.with_name(f"{uid}_tests_per_reagent_tree_pretty.png")
    pages = []
    page_count = math.ceil(len(blocks) / per_page)
    for p in range(page_count):
        chunk = blocks[p * per_page:(p + 1) * per_page]
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor=COLORS["bg"])
        axes = axes.flatten()
        for ax in axes[len(chunk):]:
            ax.axis("off")
            ax.set_facecolor(COLORS["bg"])
        for i, block in enumerate(chunk):
            render_tree(axes[i], block)
        fig.suptitle(f"{uid} – decision trees per reagent", fontsize=15, fontweight="bold", color=COLORS["title"], y=0.985)
        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])
        page_path = out_png if page_count == 1 else out_png.with_name(f"{out_png.stem}_p{p+1}{out_png.suffix}")
        fig.savefig(page_path, dpi=600, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        pages.append(page_path)
    return pages



import itertools


def outcome_for_rep(row: pd.Series, rep: str) -> str:
    winner_case = str(row["winner_case"])
    winner_rep = str(row["winner_rep"])
    if winner_case == "min":
        return "label" if rep == winner_rep else "no_label"
    if winner_case == "max":
        return "no_label" if rep == winner_rep else "label"
    return "ambiguous"


def encode_signature(vals: Sequence[str]) -> str:
    mapping = {"label": "1", "no_label": "0", "ambiguous": "?"}
    return "".join(mapping.get(v, "?") for v in vals)


def evaluate_reagent_subsets(sub: pd.DataFrame, max_subset_size: int = 2) -> Optional[dict]:
    if sub.empty:
        return None
    sub = sub.sort_values(["diff", "margin", "residue"], ascending=[False, False, True]).reset_index(drop=True)
    max_k = min(max_subset_size, len(sub))
    best = None
    best_key = None
    for k in range(1, max_k + 1):
        for idxs in itertools.combinations(range(len(sub)), k):
            sel = sub.iloc[list(idxs)].copy()
            sigs = {}
            informative_counts = 0
            total_diff = float(sel["diff"].sum())
            min_diff = float(sel["diff"].min())
            total_margin = float(sel["margin"].sum())
            for rep in REPS:
                outcomes = [outcome_for_rep(row, rep) for _, row in sel.iterrows()]
                informative_counts += sum(o != "ambiguous" for o in outcomes)
                sigs[rep] = tuple(outcomes)
            unique_count = len(set(sigs.values()))
            distinguishes_all = unique_count == len(REPS)
            residues = list(sel["residue"])
            reps_hit = sorted(set(sel["target_rep"].astype(str)))
            summary = {
                "residue_count": k,
                "residues": residues,
                "residues_str": "; ".join(residues),
                "rep1_signature": encode_signature(sigs["rep1"]),
                "rep2_signature": encode_signature(sigs["rep2"]),
                "rep3_signature": encode_signature(sigs["rep3"]),
                "distinguishes_all3": distinguishes_all,
                "unique_signature_count": unique_count,
                "total_diff": round(total_diff, 3),
                "min_diff": round(min_diff, 3),
                "total_margin": round(total_margin, 3),
                "reps_hit": "/".join(reps_hit),
                "step1_residue": residues[0] if len(residues) >= 1 else "",
                "step2_residue": residues[1] if len(residues) >= 2 else "",
            }
            key = (
                0 if distinguishes_all else 1,
                k,
                -unique_count,
                -min_diff,
                -total_diff,
                -total_margin,
                tuple(residues),
            )
            if best_key is None or key < best_key:
                best_key = key
                best = summary
    return best


def rank_reagents_for_threeway(uid: str, rules_df: pd.DataFrame, out_dir: Path,
                               max_subset_size: int = 2) -> pd.DataFrame:
    rows = []
    if rules_df.empty:
        out = pd.DataFrame(columns=[
            "rank", "reagent", "distinguishes_all3", "residue_count", "residues",
            "rep1_signature", "rep2_signature", "rep3_signature",
            "unique_signature_count", "reps_hit", "step1_residue", "step2_residue",
            "total_diff", "min_diff", "total_margin"
        ])
        out.to_csv(out_dir / f"{uid}_best_single_reagent_summary.tsv", sep='\t', index=False)
        (out_dir / f"{uid}_best_single_reagent.txt").write_text("No reagent rules available.\n")
        return out

    for reagent, sub in rules_df.groupby("reagent", sort=True):
        best = evaluate_reagent_subsets(sub, max_subset_size=max_subset_size)
        if best is None:
            continue
        rows.append({"reagent": reagent, **best})

    out = pd.DataFrame(rows)
    if out.empty:
        out.to_csv(out_dir / f"{uid}_best_single_reagent_summary.tsv", sep='\t', index=False)
        (out_dir / f"{uid}_best_single_reagent.txt").write_text("No reagent rules available.\n")
        return out

    out = out.sort_values(
        ["distinguishes_all3", "residue_count", "unique_signature_count", "min_diff", "total_diff", "total_margin", "reagent"],
        ascending=[False, True, False, False, False, False, True]
    ).reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    out.to_csv(out_dir / f"{uid}_best_single_reagent_summary.tsv", sep='\t', index=False)

    best = out.iloc[0]
    lines = [
        f"Best single reagent for {uid}",
        f"Reagent: {best['reagent']}",
        f"Distinguishes all 3 reps: {bool(best['distinguishes_all3'])}",
        f"Residues to test: {best['residues']}",
        f"Number of residues: {int(best['residue_count'])}",
        f"rep1 signature: {best['rep1_signature']}",
        f"rep2 signature: {best['rep2_signature']}",
        f"rep3 signature: {best['rep3_signature']}",
        "",
        "Signature code: 1 = label, 0 = no label, ? = ambiguous",
    ]
    (out_dir / f"{uid}_best_single_reagent.txt").write_text("\n".join(lines) + "\n")
    return out


def render_best_reagent_summary_png(summary_df: pd.DataFrame, out_png: Path, uid: str, top_k: int = 10):
    cols = ["rank", "reagent", "distinguishes_all3", "residues", "rep1_signature", "rep2_signature", "rep3_signature", "total_diff"]
    if summary_df.empty:
        fig, ax = plt.subplots(figsize=(8, 2.4))
        ax.axis("off")
        ax.text(0.5, 0.5, f"{uid}: no reagent summary available", ha="center", va="center", fontsize=12)
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return
    df = summary_df.head(top_k).copy()
    df["distinguishes_all3"] = df["distinguishes_all3"].map({True: "YES", False: "NO"})
    fig_h = max(4.2, 0.48 * (len(df) + 1.0) + 1.3)
    fig, ax = plt.subplots(figsize=(16, fig_h))
    ax.axis("off")
    display_df = df[cols].copy()
    for c in display_df.columns:
        if np.issubdtype(display_df[c].dtype, np.number):
            display_df[c] = display_df[c].round(3)
        display_df[c] = display_df[c].astype(str)
    table = ax.table(cellText=display_df.values, colLabels=display_df.columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.55)
    for j in range(len(cols)):
        cell = table[(0, j)]
        cell.set_facecolor(COLORS["header"])
        cell.get_text().set_color("white")
        cell.get_text().set_weight("bold")
        cell.set_edgecolor(COLORS["grid"])
        cell.set_linewidth(1.2)
    diff_idx = cols.index("total_diff")
    ok_idx = cols.index("distinguishes_all3")
    diffs = pd.to_numeric(df["total_diff"], errors="coerce")
    vmin = float(np.nanmin(diffs)) if len(diffs) else 0.0
    vmax = float(np.nanmax(diffs)) if len(diffs) else 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap("YlGn")
    for i in range(1, len(display_df) + 1):
        row_color = COLORS["row_alt"] if i % 2 == 0 else "white"
        for j in range(len(cols)):
            cell = table[(i, j)]
            cell.set_edgecolor(COLORS["grid"])
            cell.set_linewidth(1.0)
            if j == diff_idx:
                val = float(df.iloc[i - 1]["total_diff"])
                cell.set_facecolor(cmap(norm(val)))
                cell.get_text().set_weight("bold")
            elif j == ok_idx:
                ok = str(display_df.iloc[i - 1, j]) == "YES"
                cell.set_facecolor(COLORS["label_fill"] if ok else COLORS["nolabel_fill"])
                cell.get_text().set_weight("bold")
            else:
                cell.set_facecolor(row_color)
    fig.suptitle(f"{uid} – best single-reagent ranking", fontsize=16, weight="bold", y=0.98)
    fig.text(0.01, 0.02, "Signature code: 1 = label, 0 = no label, ? = ambiguous", fontsize=9, ha="left", va="bottom")
    fig.subplots_adjust(top=0.90, bottom=0.13, left=0.01, right=0.99)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)



def render_best_single_reagent_tree_png(summary_df: pd.DataFrame, rules_df: pd.DataFrame, out_png: Path, uid: str):
    """
    Render one clean paper-style decision tree for the top-ranked single reagent only.
    """
    if summary_df.empty:
        fig, ax = plt.subplots(figsize=(8, 2.4))
        ax.axis("off")
        ax.text(0.5, 0.5, f"{uid}: no best single reagent available", ha="center", va="center", fontsize=12)
        fig.savefig(out_png, dpi=600, bbox_inches="tight")
        plt.close(fig)
        return

    best_reagent = str(summary_df.iloc[0]["reagent"])
    sub = rules_df[rules_df["reagent"] == best_reagent].copy()
    if sub.empty:
        fig, ax = plt.subplots(figsize=(8, 2.4))
        ax.axis("off")
        ax.text(0.5, 0.5, f"{uid}: no rules found for best reagent {best_reagent}", ha="center", va="center", fontsize=12)
        fig.savefig(out_png, dpi=600, bbox_inches="tight")
        plt.close(fig)
        return

    tree = build_tree_for_reagent(sub, max_steps=2)
    if tree is None:
        fig, ax = plt.subplots(figsize=(8, 2.4))
        ax.axis("off")
        ax.text(0.5, 0.5, f"{uid}: could not build a tree for {best_reagent}", ha="center", va="center", fontsize=12)
        fig.savefig(out_png, dpi=600, bbox_inches="tight")
        plt.close(fig)
        return

    sig1 = str(summary_df.iloc[0].get("rep1_signature", ""))
    sig2 = str(summary_df.iloc[0].get("rep2_signature", ""))
    sig3 = str(summary_df.iloc[0].get("rep3_signature", ""))
    residues = str(summary_df.iloc[0].get("residues", ""))
    distinguishes_all3 = bool(summary_df.iloc[0].get("distinguishes_all3", False))

    def _pick_branch(branch_name: str):
        for b in tree.get("branches", []):
            if b.get("branch") == branch_name:
                return b
        return {"step2_residue": "", "step2_if_label": [], "step2_if_no_label": [], "decision": []}

    label_branch = _pick_branch("label")
    no_label_branch = _pick_branch("no_label")

    block = {
        "reagent": best_reagent,
        "step1_res": tree.get("step1_residue", ""),
        "step1_lab": fmt_rep_set(tree.get("step1_if_label", [])),
        "step1_nlab": fmt_rep_set(tree.get("step1_if_no_label", [])),
        "step2_label_res": label_branch.get("step2_residue", ""),
        "step2_label_lab": fmt_rep_set(label_branch.get("step2_if_label", [])),
        "step2_label_nlab": fmt_rep_set(label_branch.get("step2_if_no_label", [])),
        "step2_no_label_res": no_label_branch.get("step2_residue", ""),
        "step2_no_label_lab": fmt_rep_set(no_label_branch.get("step2_if_label", [])),
        "step2_no_label_nlab": fmt_rep_set(no_label_branch.get("step2_if_no_label", [])),
        "label_branch": fmt_rep_set(label_branch.get("decision", [])),
        "no_label_branch": fmt_rep_set(no_label_branch.get("decision", [])),
    }

    fig, ax = plt.subplots(figsize=(10.5, 7.8), facecolor=COLORS["bg"])
    render_tree(ax, block)
    fig.suptitle(f"{uid} – best single-reagent decision tree", fontsize=16, fontweight="bold",
                 color=COLORS["title"], y=0.985)

    footer_lines = [
        f"Best reagent: {best_reagent}",
        f"Residues used: {residues}",
        f"Distinguishes all 3 reps: {'YES' if distinguishes_all3 else 'NO'}",
        f"rep1 signature = {sig1}    rep2 signature = {sig2}    rep3 signature = {sig3}",
        "Signature code: 1 = label, 0 = no label, ? = ambiguous",
    ]
    fig.text(0.03, 0.03, "\n".join(footer_lines), ha="left", va="bottom",
             fontsize=10, color=COLORS["text"])

    fig.subplots_adjust(top=0.92, bottom=0.16, left=0.03, right=0.97)
    fig.savefig(out_png, dpi=600, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def write_best_single_reagent_outputs(uid: str, summary_df: pd.DataFrame, rules_df: pd.DataFrame, out_dir: Path):
    """
    Keep all existing outputs, but also write one final user-facing answer:
    - <uid>_final_recommendation.txt
    - <uid>_final_recommendation.png
    - <uid>_best_single_reagent_residues.tsv
    """
    txt_path = out_dir / f"{uid}_final_recommendation.txt"
    png_path = out_dir / f"{uid}_final_recommendation.png"
    best_rows_path = out_dir / f"{uid}_best_single_reagent_residues.tsv"

    if summary_df.empty:
        txt_path.write_text(f"No best single reagent recommendation could be generated for {uid}.\n")
        fig, ax = plt.subplots(figsize=(8, 2.4))
        ax.axis("off")
        ax.text(0.5, 0.5, f"{uid}: no final recommendation available", ha="center", va="center", fontsize=12)
        fig.savefig(png_path, dpi=600, bbox_inches="tight")
        plt.close(fig)
        pd.DataFrame().to_csv(best_rows_path, sep="\t", index=False)
        return txt_path, png_path, best_rows_path

    best = summary_df.iloc[0]
    best_reagent = str(best["reagent"])
    residues = [x.strip() for x in str(best.get("residues", "")).split(";") if x.strip()]
    sub = rules_df[(rules_df["reagent"] == best_reagent) & (rules_df["residue"].isin(residues))].copy()
    if sub.empty:
        sub = rules_df[rules_df["reagent"] == best_reagent].copy()

    preferred = {res: i for i, res in enumerate(residues)}
    if not sub.empty:
        sub["_res_order"] = sub["residue"].map(lambda x: preferred.get(str(x), 999))
        sub = sub.sort_values(["_res_order", "diff", "margin", "residue"], ascending=[True, False, False, True]).drop(columns=["_res_order"])
    sub.to_csv(best_rows_path, sep="\t", index=False)

    tree = build_tree_for_reagent(sub, max_steps=2) if not sub.empty else None

    lines = [
        f"Recommended single-reagent strategy for {uid}",
        f"Best reagent: {best_reagent}",
        f"Distinguishes all 3 reps: {'YES' if bool(best.get('distinguishes_all3', False)) else 'NO'}",
        f"Residues to test: {best.get('residues', '')}",
        f"rep1 signature: {best.get('rep1_signature', '')}",
        f"rep2 signature: {best.get('rep2_signature', '')}",
        f"rep3 signature: {best.get('rep3_signature', '')}",
        "",
        "Decision rule:",
    ]
    if tree is None:
        lines.append("Could not build a decision tree for the top-ranked reagent.")
    else:
        lines.append(f"1. Test {tree['step1_residue']}.")
        if len(tree["step1_if_no_label"]) == 1:
            lines.append(f"   - If NOT labeled -> {tree['step1_if_no_label'][0]}")
        else:
            lines.append(f"   - If NOT labeled -> {fmt_rep_set(tree['step1_if_no_label'])}")
        if len(tree["step1_if_label"]) == 1:
            lines.append(f"   - If labeled -> {tree['step1_if_label'][0]}")
        else:
            label_branch = next((b for b in tree["branches"] if b.get("branch") == "label"), None)
            if label_branch and label_branch.get("step2_residue"):
                lines.append(f"   - If labeled, test {label_branch['step2_residue']}:")
                lines.append(f"       • labeled -> {fmt_rep_set(label_branch['step2_if_label'])}")
                lines.append(f"       • not labeled -> {fmt_rep_set(label_branch['step2_if_no_label'])}")
            else:
                lines.append(f"   - If labeled -> {fmt_rep_set(tree['step1_if_label'])}")
        # include no-label branch step2 if present
        no_label_branch = next((b for b in tree["branches"] if b.get("branch") == "no_label"), None)
        if no_label_branch and no_label_branch.get("step2_residue"):
            lines.append(f"2. If {tree['step1_residue']} is NOT labeled, test {no_label_branch['step2_residue']}:")
            lines.append(f"       • labeled -> {fmt_rep_set(no_label_branch['step2_if_label'])}")
            lines.append(f"       • not labeled -> {fmt_rep_set(no_label_branch['step2_if_no_label'])}")

    lines += ["", "Signature code: 1 = label, 0 = no label, ? = ambiguous"]
    txt_path.write_text("\n".join(lines) + "\n")

    # Figure
    fig, ax = plt.subplots(figsize=(11, 8.5), facecolor=COLORS["bg"])
    ax.axis("off")
    ax.set_facecolor(COLORS["bg"])

    title = f"{uid} – final recommendation"
    ax.text(0.5, 0.96, title, ha="center", va="top",
            fontsize=18, fontweight="bold", color=COLORS["title"], transform=ax.transAxes)

    info_lines = [
        f"Best reagent: {best_reagent}",
        f"Residues: {best.get('residues', '')}",
        f"Distinguishes all 3 reps: {'YES' if bool(best.get('distinguishes_all3', False)) else 'NO'}",
        f"rep1 = {best.get('rep1_signature', '')}    rep2 = {best.get('rep2_signature', '')}    rep3 = {best.get('rep3_signature', '')}",
    ]
    draw_round_box(ax, (0.5, 0.83), 0.82, 0.12, "\n".join(info_lines),
                   fc=COLORS["panel"], ec=COLORS["line"], fontsize=11, weight="bold")

    if tree is None:
        ax.text(0.5, 0.52, "Could not build a decision tree for the top-ranked reagent.",
                ha="center", va="center", fontsize=13, color=COLORS["text"], transform=ax.transAxes)
    else:
        # Shifted tree layout slightly downward
        step1 = (0.5, 0.63)
        left_decision = (0.23, 0.19)
        middle_decision = (0.50, 0.19)
        right_decision = (0.77, 0.19)
        step2_left = (0.30, 0.39)
        step2_right = (0.70, 0.39)

        draw_round_box(ax, step1, 0.28, 0.09, f"Step 1\n{tree['step1_residue']}",
                       COLORS["step_fill"], COLORS["line"], fontsize=10, weight="bold")
        draw_connector(ax, (0.44, 0.585), (0.30, 0.47), COLORS["labeled"])
        draw_connector(ax, (0.56, 0.585), (0.70, 0.47), COLORS["not_labeled"])
        draw_branch_label(ax, (0.29, 0.515), "LABELED", COLORS["labeled"], COLORS["label_fill"])
        draw_branch_label(ax, (0.71, 0.515), "NOT LABELED", COLORS["not_labeled"], COLORS["nolabel_fill"])

        # label branch
        label_branch = next((b for b in tree["branches"] if b.get("branch") == "label"), None)
        if label_branch and label_branch.get("step2_residue"):
            draw_round_box(ax, step2_left, 0.26, 0.09, f"Step 2\n{label_branch['step2_residue']}",
                           COLORS["step_fill"], COLORS["line"], fontsize=10, weight="bold")
            draw_connector(ax, (0.26, 0.345), (0.20, 0.24), COLORS["labeled"])
            draw_connector(ax, (0.34, 0.345), (0.43, 0.24), COLORS["not_labeled"])
            draw_branch_label(ax, (0.18, 0.285), "LABELED", COLORS["labeled"], COLORS["label_fill"])
            draw_branch_label(ax, (0.42, 0.285), "NOT LABELED", COLORS["not_labeled"], COLORS["nolabel_fill"])
            draw_round_box(ax, left_decision, 0.22, 0.08, fmt_rep_set(label_branch["step2_if_label"]),
                           COLORS["decision_fill"], COLORS["line"], fontsize=10)
            draw_round_box(ax, middle_decision, 0.22, 0.08, fmt_rep_set(label_branch["step2_if_no_label"]),
                           COLORS["decision_fill"], COLORS["line"], fontsize=10)
        else:
            draw_round_box(ax, step2_left, 0.28, 0.08, fmt_rep_set(tree["step1_if_label"]),
                           COLORS["decision_fill"], COLORS["line"], fontsize=10)

        # no-label branch
        no_label_branch = next((b for b in tree["branches"] if b.get("branch") == "no_label"), None)
        if no_label_branch and no_label_branch.get("step2_residue"):
            draw_round_box(ax, step2_right, 0.26, 0.09, f"Step 2\n{no_label_branch['step2_residue']}",
                           COLORS["step_fill"], COLORS["line"], fontsize=10, weight="bold")
            draw_connector(ax, (0.66, 0.345), (0.58, 0.24), COLORS["labeled"])
            draw_connector(ax, (0.74, 0.345), (0.80, 0.24), COLORS["not_labeled"])
            draw_branch_label(ax, (0.58, 0.285), "LABELED", COLORS["labeled"], COLORS["label_fill"])
            draw_branch_label(ax, (0.82, 0.285), "NOT LABELED", COLORS["not_labeled"], COLORS["nolabel_fill"])
            draw_round_box(ax, middle_decision, 0.22, 0.08, fmt_rep_set(no_label_branch["step2_if_label"]),
                           COLORS["decision_fill"], COLORS["line"], fontsize=10)
            draw_round_box(ax, right_decision, 0.22, 0.08, fmt_rep_set(no_label_branch["step2_if_no_label"]),
                           COLORS["decision_fill"], COLORS["line"], fontsize=10)
        else:
            draw_round_box(ax, step2_right, 0.28, 0.08, fmt_rep_set(tree["step1_if_no_label"]),
                           COLORS["decision_fill"], COLORS["line"], fontsize=10)

    ax.text(0.03, 0.04,
            "Signature code: 1 = label, 0 = no label, ? = ambiguous",
            ha="left", va="bottom", fontsize=10, color=COLORS["text"], transform=ax.transAxes)
    fig.savefig(png_path, dpi=600, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return txt_path, png_path, best_rows_path

def main():
    ap = argparse.ArgumentParser(description="Self-contained Step 4 CL-MS decision script.")
    ap.add_argument("--uniprot", required=True)
    ap.add_argument("--nc-tsv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--labels-source", default="", help="Optional TSV/CSV: resname, labels, label_non_specific.")
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--min-winner-diff", type=float, default=0.0)
    ap.add_argument("--min-margin", type=float, default=0.0)
    ap.add_argument("--tie-eps", type=float, default=1e-9)
    ap.add_argument("--max-steps", type=int, default=2, choices=[1, 2])
    ap.add_argument("--min-rows", type=int, default=2)
    args = ap.parse_args()

    uid = args.uniprot
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nc_df = read_nc_table(Path(args.nc_tsv))
    reagent_map = read_reagent_map(Path(args.labels_source) if args.labels_source.strip() else None)

    rank_df = compute_rank_table(nc_df, tie_eps=args.tie_eps)
    rank_df = attach_reagent_columns(rank_df, reagent_map)
    rank_df.to_csv(out_dir / f"{uid}_cl_rank_all_residues.tsv", sep="\t", index=False)
    rank_df[rank_df["winner_case"] == "min"].sort_values(["winner_diff", "margin"], ascending=[False, False]).to_csv(
        out_dir / f"{uid}_cl_rank_exposed.tsv", sep="\t", index=False
    )
    rank_df[rank_df["winner_case"] == "max"].sort_values(["winner_diff", "margin"], ascending=[False, False]).to_csv(
        out_dir / f"{uid}_cl_rank_buried.tsv", sep="\t", index=False
    )

    top_df = build_top_tables(uid, rank_df, out_dir, args.top_n,
                              min_winner_diff=args.min_winner_diff,
                              min_margin=args.min_margin)
    render_top_table_png(top_df, out_dir / f"{uid}_top{args.top_n}_table.png", uid, args.top_n)

    reagent_residue_df = build_reagent_residue_tables(uid, rank_df, out_dir)

    rules_df = build_reagent_rules_table(top_df)
    rules_path = out_dir / f"{uid}_reagent_targets.tsv"
    rules_df.to_csv(rules_path, sep="\t", index=False)

    summary_df = rank_reagents_for_threeway(uid, rules_df, out_dir, max_subset_size=2)
    render_best_reagent_summary_png(summary_df, out_dir / f"{uid}_best_single_reagent_summary.png", uid, top_k=10)
    render_best_single_reagent_tree_png(summary_df, rules_df, out_dir / f"{uid}_best_single_reagent_tree.png", uid)
    final_txt, final_png, final_rows = write_best_single_reagent_outputs(uid, summary_df, rules_df, out_dir)

    tests_txt, tests_tsv = design_tests_from_rules(uid, rules_df, out_dir,
                                                   max_steps=args.max_steps,
                                                   min_rows=args.min_rows)
    tree_pages = render_tests_per_reagent_tree_pretty(tests_txt, per_page=4)

    print("Wrote:")
    print(" ", out_dir / f"{uid}_cl_rank_all_residues.tsv")
    print(" ", out_dir / f"{uid}_cl_rank_exposed.tsv")
    print(" ", out_dir / f"{uid}_cl_rank_buried.tsv")
    for rep in REPS:
        print(" ", out_dir / f"{uid}_{rep}_top{args.top_n}.tsv")
    print(" ", out_dir / f"{uid}_top{args.top_n}_all_reps.tsv")
    print(" ", out_dir / f"{uid}_top{args.top_n}_table.png")
    print(" ", out_dir / f"{uid}_reagent_residue_table_all.tsv")
    print(" ", rules_path)
    print(" ", out_dir / f"{uid}_best_single_reagent_summary.tsv")
    print(" ", out_dir / f"{uid}_best_single_reagent_summary.png")
    print(" ", out_dir / f"{uid}_best_single_reagent.txt")
    print(" ", tests_tsv)
    print(" ", tests_txt)
    for p in tree_pages:
        print(" ", p)

if __name__ == "__main__":
    main()
