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
    "EDC/GEE",
    "Koshland's reagent (HNB bromide)",
    "O-nitrophenylsulfenyl chloride",
    "Tetranitromethane",
    "Iodine",
]
NONSPEC_ORDER = ["OH-medium"]
NONSPEC = set(NONSPEC_ORDER)

# Published broad OH coverage uses a single medium category only.
OH_MEDIUM = {"TRP", "TYR", "PHE", "HIS", "LEU", "ILE", "ARG", "LYS", "VAL", "PRO"}

SPECIFIC = {
    "HIS": ["DEPC", "N-bromosuccinimide (NBS)", "Iodine"],
    "LYS": ["DEPC", "N-acetylimidazole", "Acetic anhydride",
            "Succinic anhydride", "Maleic anhydride",
            "S-methylthiocacetimidate"],
    "CYS": ["DEPC", "Iodoacetamide/iodoacetate", "Acryloyl"],
    "SER": ["DEPC"],
    "THR": ["DEPC"],
    "TYR": ["DEPC", "N-acetylimidazole", "N-bromosuccinimide (NBS)",
            "Tetranitromethane", "Iodine"],
    "ASP": ["EDC/GEE"],
    "GLU": ["EDC/GEE"],
    "ARG": ["Phenylglyoxal", "p-hydroxyphenylglyoxal", "2,3-butanedione",
            "1,2-cyclohexanedione", "Methylglyoxal", "Kethoxal"],
    "TRP": ["N-bromosuccinimide (NBS)", "Koshland's reagent (HNB bromide)",
            "O-nitrophenylsulfenyl chloride"],
}

S2_ROWS = [
    ("section", "Residue-selective", "", "", ""),
    ("row", "DEPC", "His, Lys, Cys, Ser, Thr, Tyr",
     "Published DEPC reactivity",
     "Mendoza & Vachet 2009; Limpikirati et al. 2019"),
    ("row", "EDC/GEE", "Asp, Glu",
     "Carboxyl footprinting",
     "Zhang et al. 2011; Kaur et al. 2015"),
    ("row", "N-acetylimidazole", "Tyr, Lys",
     "Tyr preferred; Lys also acetylated",
     "Riordan et al. 1965"),
    ("row", "2,3-Butanedione", "Arg",
     "Guanidino-group labeling",
     "Yankeelov 1970"),
    ("row", "1,2-Cyclohexanedione", "Arg",
     "Guanidino-group labeling",
     "Patthy & Smith 1975"),
    ("row", "Phenylglyoxal", "Arg",
     "Guanidino-group labeling",
     "Takahashi 1968"),
    ("row", "Methylglyoxal", "Arg",
     "Guanidino-group labeling",
     "Lo et al. 1994"),
    ("row", "p-Hydroxyphenylglyoxal", "Arg",
     "Guanidino-group labeling",
     "Yamasaki et al. 1981"),
    ("row", "Kethoxal", "Arg",
     "Guanidino-group labeling",
     "Litt & Hancock 1987"),
    ("row", "Acetic anhydride", "Lys",
     "Lysine acetylation",
     "Means & Feeney 1971"),
    ("row", "Succinic anhydride", "Lys",
     "Lysine succinylation",
     "Klotz 1967"),
    ("row", "Maleic anhydride", "Lys",
     "Lysine maleylation",
     "Butler et al. 1969"),
    ("row", "S-methylthioacetimidate", "Lys",
     "Lysine amidination",
     "Hunter & Ludwig 1962"),
    ("row", "Iodoacetamide / iodoacetate", "Cys",
     "Cys alkylation",
     "Crestfield et al. 1963"),
    ("row", "Acryloyl", "Cys",
     "Cys Michael addition",
     "Friedman et al. 1965"),
    ("row", "N-Bromosuccinimide", "Trp, Tyr, His",
     "Trp primary; Tyr/His oxidation reported",
     "Patchornik et al. 1958; Spande & Witkop 1967"),
    ("row", "HNB bromide", "Trp",
     "Trp-selective",
     "Koshland et al. 1964"),
    ("row", "NPS-Cl", "Trp",
     "Trp-selective",
     "Scoffone et al. 1968"),
    ("row", "Tetranitromethane", "Tyr",
     "Tyr nitration",
     "Sokolovsky et al. 1966"),
    ("row", "Iodine", "Tyr, His",
     "Aromatic iodination",
     "Hughes & Straessle 1950"),
    ("section", "Broadly reactive (OH)", "", "", ""),
    ("row", "Hydroxyl radical, medium",
     "Trp, Tyr, Phe, His, Leu, Ile, Arg, Lys, Val, Pro",
     "Single OH category used in this work",
     "User-defined panel"),
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
