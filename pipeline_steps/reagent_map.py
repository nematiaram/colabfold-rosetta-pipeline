#!/usr/bin/env python3
"""Canonical residue–reagent lookup used for reporter assignment.

Residue-selective sets are published experimental specificities, not
encyclopedia side-reaction lists.

Hydroxyl-radical high/medium/low are operational bins defined in this
study from the Xu & Chance (2005) intrinsic-reactivity order, with
cutoffs after Arg and after Glu. They are not categories defined by
Xu & Chance.

CF3 and diazirine are mapped to all 20 standard amino acids as an
operational pipeline assumption (broad labeling), not as a claim that
Cheng et al. (2017) demonstrated all 20 or that residue reactivities
are experimentally uniform.
"""
AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

# Preference among residue-selective reagents when several match.
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
NONSPEC_ORDER = ["OH-high", "OH-medium", "OH-low", "diazirine", "CF3"]
NONSPEC = set(NONSPEC_ORDER)

# Operational bins from the Xu & Chance 2005 Anal. Chem. 77:4549 order:
#   Cys > Met > Trp > Tyr > Phe > His > Leu, Ile > Arg, Lys, Val
#   > Ser, Thr, Pro > Gln, Glu > Asp, Asn > Ala > Gly
# High  = Cys through Arg (cutoff after Arg)
# Medium = Lys through Glu (cutoff after Glu)
# Low   = Asp through Gly
# These high/medium/low labels are this study's bins, not Xu & Chance categories.
OH_HIGH = {"CYS", "MET", "TRP", "TYR", "PHE", "HIS", "LEU", "ILE", "ARG"}
OH_MEDIUM = {"LYS", "VAL", "SER", "THR", "PRO", "GLN", "GLU"}
OH_LOW = {"ASP", "ASN", "ALA", "GLY"}

# Residue-selective: published experimental targets (typical CL/FP use).
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

# Table S2 display (one row per reagent).
S2_ROWS = [
    ("section", "Residue-selective", "", "", ""),
    ("row", "DEPC", "His, Lys, Cys, Ser, Thr, Tyr",
     "Published DEPC reactivity",
     "Mendoza & Vachet 2009; Limpikirati et al. 2019"),
    ("row", "EDC/GEE", "Asp, Glu",
     "Carboxyl footprinting",
     "Zhang et al. 2012; Kaur et al. 2015"),
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
    ("section", "Broadly reactive", "", "", ""),
    ("row", "Hydroxyl radical, high",
     "Cys, Met, Trp, Tyr, Phe, His, Leu, Ile, Arg",
     "Operational reactivity groups based on the Xu & Chance (2005) intrinsic-reactivity order",
     "Xu & Chance 2005"),
    ("row", "Hydroxyl radical, medium",
     "Lys, Val, Ser, Thr, Pro, Gln, Glu",
     "Operational reactivity groups based on the Xu & Chance (2005) intrinsic-reactivity order",
     "Xu & Chance 2005"),
    ("row", "Hydroxyl radical, low",
     "Asp, Asn, Ala, Gly",
     "Operational reactivity groups based on the Xu & Chance (2005) intrinsic-reactivity order",
     "Xu & Chance 2005"),
    ("row", "Diazirine",
     "All 20 standard amino acids",
     "Broad diazirine/carbene labeling; all-20 mapping used as an operational pipeline assumption",
     "Richards et al. 2016"),
    ("row", "CF3",
     "All 20 standard amino acids",
     "Broad CF3 labeling; all-20 mapping used as an operational pipeline assumption",
     "Cheng et al. 2017"),
]


def normalize_aa(resname: str) -> str:
    s = str(resname).strip().upper()
    if s in AA3_TO_1:
        return s
    if len(s) == 1 and s in {v: k for k, v in AA3_TO_1.items()}:
        inv = {v: k for k, v in AA3_TO_1.items()}
        return inv[s]
    return s


def specific_for(aa: str):
    aa = normalize_aa(aa)
    names = SPECIFIC.get(aa, [])
    return [r for r in REAGENT_ORDER if r in names]


def nonspec_for(aa: str):
    aa = normalize_aa(aa)
    out = []
    if aa in OH_HIGH:
        out.append("OH-high")
    elif aa in OH_MEDIUM:
        out.append("OH-medium")
    elif aa in OH_LOW:
        out.append("OH-low")
    if aa in AA3_TO_1:
        out.append("diazirine")
        out.append("CF3")
    return out


def assign(resname: str) -> dict:
    spec = specific_for(resname)
    ns = nonspec_for(resname)
    if spec:
        preferred, tier, has = spec[0], "specific", True
    elif ns:
        preferred, tier, has = ns[0], "non_specific", False
    else:
        preferred, tier, has = "", "none", False
    reagents = spec + ns
    labels = "; ".join(f"*{REAGENT_ORDER.index(r)+1}" for r in spec)
    return {
        "preferred_reagent": preferred,
        "reagent_tier": tier,
        "has_specific": has,
        "reagents": "; ".join(reagents),
        "labels": labels,
        "label_non_specific": "; ".join(ns),
    }
