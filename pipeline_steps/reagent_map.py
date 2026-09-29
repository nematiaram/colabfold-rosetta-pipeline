#!/usr/bin/env python3
"""Canonical residue-reagent lookup used for reporter assignment.

For the published panel, hydroxyl-radical coverage is represented by a single
OH-medium category applied only to Trp, Tyr, Phe, His, Leu, Ile, Arg, Lys,
Val, and Pro. Diazirine and CF3 are excluded from the published panel.
"""

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

# Order matches the SI residue-selective table, then used for *N labels
# and preferred_reagent (first matching name in this list).
REAGENT_ORDER = [
    "DEPC",
    "EDC/GEE",
    "N-acetylimidazole",
    "N-bromosuccinimide (NBS)",
    "Iodine",
    "Phenylglyoxal",
    "p-hydroxyphenylglyoxal",
    "2,3-butanedione",
    "1,2-cyclohexanedione",
    "Methylglyoxal",
    "Kethoxal",
    "Acetic anhydride",
    "Succinic anhydride",
    "Maleic anhydride",
    "S-methylthioacetimidate",
    "Iodoacetamide/iodoacetate",
    "2-Hydroxy-5-nitrobenzyl bromide (HNB)",
    "2-Nitrophenylsulfenyl chloride (NPS-Cl)",
    "Tetranitromethane",
]
NONSPEC_ORDER = ["OH-medium"]
NONSPEC = set(NONSPEC_ORDER)

# Published broad OH coverage uses a single medium category only.
OH_MEDIUM = {"TRP", "TYR", "PHE", "HIS", "LEU", "ILE", "ARG", "LYS", "VAL", "PRO"}

# Residues used column of the SI table (not side-reaction footnotes).
SPECIFIC = {
    "HIS": ["DEPC", "N-bromosuccinimide (NBS)", "Iodine"],
    "LYS": ["DEPC", "N-acetylimidazole", "Acetic anhydride",
            "Succinic anhydride", "Maleic anhydride",
            "S-methylthioacetimidate"],
    "CYS": ["DEPC", "Iodoacetamide/iodoacetate"],
    "SER": ["DEPC"],
    "THR": ["DEPC"],
    "TYR": ["DEPC", "N-acetylimidazole", "N-bromosuccinimide (NBS)",
            "Iodine", "Tetranitromethane"],
    "ASP": ["EDC/GEE"],
    "GLU": ["EDC/GEE"],
    "ARG": ["Phenylglyoxal", "p-hydroxyphenylglyoxal", "2,3-butanedione",
            "1,2-cyclohexanedione", "Methylglyoxal", "Kethoxal"],
    "TRP": ["N-bromosuccinimide (NBS)",
            "2-Hydroxy-5-nitrobenzyl bromide (HNB)",
            "2-Nitrophenylsulfenyl chloride (NPS-Cl)"],
}

# SI table: reagent, residues used, reported specificity, caveats, example.
S2_ROWS = [
    ("section", "Residue-selective reagents (tier specific)", "", "", "", ""),
    ("row", "Diethylpyrocarbonate (DEPC)", "His, Lys, Cys, Ser, Thr, Tyr",
     "His primary; Lys, Tyr, Cys, Ser, Thr, and Arg as side reactions",
     "Hydrolyzes in water (half-life about 9 min at pH 7); His adducts reverse over hours",
     "Mendoza and Vachet, Anal. Chem. 2008"),
    ("row", "1-Ethyl-3-(3-dimethylaminopropyl)carbodiimide with glycine ethyl ester (EDC/GEE)", "Asp, Glu",
     "Asp, Glu, C-terminal carboxylate",
     "",
     "Kaur et al., mAbs 2015"),
    ("row", "N-Acetylimidazole (NAI)", "Tyr, Lys",
     "Tyr primary; Lys and Ser less readily",
     "",
     "Zappacosta et al., Protein Sci. 1997"),
    ("row", "N-Bromosuccinimide (NBS)", "Trp, Tyr, His",
     "Trp primary; Tyr, His, Arg, amines, and thiols occasionally",
     "Halide-free conditions required",
     "Ali et al., J. Biol. Chem. 1995"),
    ("row", "Iodine", "Tyr, His",
     "Tyr primary, His significant; also oxidizes Met, Cys, and Trp",
     "",
     "Rosenfeld et al., Protein Sci. 1993"),
    ("row", "Phenylglyoxal (PG)", "Arg",
     "Arg; amines in the absence of borate",
     "Borate stabilizes the Arg adduct and suppresses amine side reactions",
     "Krell et al., FEBS Lett. 1995"),
    ("row", "p-Hydroxyphenylglyoxal (pHPG)", "Arg",
     "Arg",
     "As for PG",
     "Carven and Stern, Biochemistry 2005"),
    ("row", "2,3-Butanedione (BD)", "Arg",
     "Arg; minor Lys and His",
     "As for PG",
     "Leitner et al., Rapid Commun. Mass Spectrom. 2007"),
    ("row", "1,2-Cyclohexanedione (CHD)", "Arg",
     "Arg; amines in the absence of borate",
     "As for PG",
     "Suckau et al., Proc. Natl. Acad. Sci. U.S.A. 1992"),
    ("row", "Methylglyoxal (MG)", "Arg",
     "Arg primary; Lys considerable; Cys",
     "As for PG",
     "Chen et al., Chem. Res. Toxicol. 2015"),
    ("row", "Kethoxal", "Arg",
     "Arg in proteins (originally a guanine-specific RNA probe)",
     "",
     "Akinsiku et al., J. Mass Spectrom. 2005"),
    ("row", "Acetic anhydride (Ac2O)", "Lys",
     "Lys, N-terminal amine; transient Tyr, Ser, and Thr acylation",
     "",
     "Gong et al., Biochemistry 2011"),
    ("row", "Succinic anhydride (SucAnh)", "Lys",
     "Lys, N-terminal amine",
     "",
     "Glocker et al., Bioconjugate Chem. 1994"),
    ("row", "Maleic anhydride (MalAnh)", "Lys",
     "Lys, N-terminal amine",
     "Adduct is reversible at acidic pH",
     "Ehrhard et al., Biochemistry 1996"),
    ("row", "S-Methylthioacetimidate (SMTA)", "Lys",
     "Lys, N-terminal amine (charge-retaining amidination)",
     "",
     "Jaffee and Reilly, Anal. Chem. 2012"),
    ("row", "Iodoacetamide or iodoacetate (IAA)", "Cys",
     "Cys primary; His, Met, and N-terminus at high pH or reagent excess",
     "",
     "Weerapana et al., Nature 2010"),
    ("row", "2-Hydroxy-5-nitrobenzyl bromide (HNB; Koshland's reagent)", "Trp",
     "Trp; Cys below pH 4 and Tyr at alkaline pH",
     "Light- and hydrolysis-sensitive",
     "Strohalm et al., Biochem. Biophys. Res. Commun. 2004"),
    ("row", "2-Nitrophenylsulfenyl chloride (NPS-Cl)", "Trp",
     "Trp; Cys sulfenylated to a similar extent",
     "",
     "Chang et al., J. Protein Chem. 1995"),
    ("row", "Tetranitromethane (TNM)", "Tyr",
     "Tyr; His, Met, and Trp at high excess or pH 8 and above",
     "",
     "Gong et al., Biochemistry 2011"),
    ("section", "Broadly reactive reagent (tier non-specific)", "", "", "", ""),
    ("row", "Hydroxyl radical, medium-reactivity category (OH-medium)",
     "Trp, Tyr, Phe, His, Leu, Ile, Arg, Lys, Val, Pro",
     "Cys and Met react faster with the hydroxyl radical but are outside this category",
     "A single medium-reactivity category is used in the panel",
     "Hambly and Gross, J. Am. Soc. Mass Spectrom. 2005"),
]


def normalize_aa(resname):
    s = str(resname).strip().upper()
    if s in AA3_TO_1:
        return s
    if len(s) == 1:
        inv = {v: k for k, v in AA3_TO_1.items()}
        if s in inv:
            return inv[s]
    return s


def specific_for(aa):
    aa = normalize_aa(aa)
    names = SPECIFIC.get(aa, [])
    return [r for r in REAGENT_ORDER if r in names]


def nonspec_for(aa):
    aa = normalize_aa(aa)
    out = []
    if aa in OH_MEDIUM:
        out.append("OH-medium")
    return out


def assign(resname):
    spec = specific_for(resname)
    ns = nonspec_for(resname)
    if spec:
        preferred, tier, has = spec[0], "specific", True
    elif ns:
        preferred, tier, has = ns[0], "non_specific", False
    else:
        preferred, tier, has = "", "none", False
    reagents = spec + ns
    labels = "; ".join("*{}".format(REAGENT_ORDER.index(r) + 1) for r in spec)
    return {
        "preferred_reagent": preferred,
        "reagent_tier": tier,
        "has_specific": has,
        "reagents": "; ".join(reagents),
        "labels": labels,
        "label_non_specific": "; ".join(ns),
    }


def _parse_residue_token(token):
    """Normalize a user residue token (1-letter or 3-letter) to AA3."""
    aa = normalize_aa(token)
    if aa not in AA3_TO_1:
        raise ValueError(
            "Unknown residue type %r (use 1-letter or 3-letter amino-acid codes)"
            % (token,)
        )
    return aa


def parse_custom_reagent_entry(raw):
    """Validate one custom reagent dict -> {name, residues: [AA3,...]}."""
    if not isinstance(raw, dict):
        raise ValueError("Each custom reagent must be an object with name and residues")
    name = str(raw.get("name", "")).strip()
    if not name:
        raise ValueError("Custom reagent is missing a non-empty 'name'")
    residues_raw = raw.get("residues", None)
    if residues_raw is None:
        raise ValueError("Custom reagent %r is missing 'residues'" % name)
    if isinstance(residues_raw, str):
        parts = [p.strip() for p in residues_raw.replace(";", ",").split(",")]
        residues_raw = [p for p in parts if p]
    if not isinstance(residues_raw, (list, tuple)) or not residues_raw:
        raise ValueError(
            "Custom reagent %r must list one or more residues" % name
        )
    residues = []
    seen = set()
    for tok in residues_raw:
        aa = _parse_residue_token(tok)
        if aa not in seen:
            residues.append(aa)
            seen.add(aa)
    return {"name": name, "residues": residues}


def load_custom_reagents(path_or_list):
    """Load custom reagents from a JSON path or an in-memory list/dict.

    Accepted JSON shapes:
      [{"name": "MyReagent", "residues": ["LYS", "CYS"]}, ...]
      {"custom_reagents": [ ... ]}
      {"name": "...", "residues": [...]}   # single entry
    """
    import json
    from pathlib import Path

    if path_or_list is None:
        return []
    if isinstance(path_or_list, (str, Path)):
        path = Path(path_or_list)
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = path_or_list

    if isinstance(data, dict):
        if "custom_reagents" in data:
            data = data["custom_reagents"]
        elif "name" in data and "residues" in data:
            data = [data]
        else:
            raise ValueError(
                "Custom reagents JSON must be a list, a single "
                "{name, residues} object, or {custom_reagents: [...]}"
            )
    if not isinstance(data, list):
        raise ValueError("Custom reagents must be a JSON list")
    return [parse_custom_reagent_entry(entry) for entry in data]


def apply_custom_reagents(entries):
    """Additively merge custom reagents into REAGENT_ORDER and SPECIFIC.

    Built-in reagents are never removed. Custom names are appended to
    REAGENT_ORDER (after built-ins) and to each targeted residue's SPECIFIC
    list. Preferred reagent therefore stays the first built-in match when
    one exists; otherwise the first matching custom reagent is preferred.
    Returns the normalized entry list that was applied.
    """
    global REAGENT_ORDER, SPECIFIC

    if not entries:
        return []

    applied = []
    for entry in entries:
        parsed = parse_custom_reagent_entry(entry) if "residues" in entry else entry
        name = parsed["name"]
        residues = parsed["residues"]
        if name not in REAGENT_ORDER:
            REAGENT_ORDER = list(REAGENT_ORDER) + [name]
        for aa in residues:
            current = list(SPECIFIC.get(aa, []))
            if name not in current:
                current.append(name)
            SPECIFIC[aa] = current
        applied.append({"name": name, "residues": list(residues)})
    return applied
