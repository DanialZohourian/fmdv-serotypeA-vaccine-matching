# -*- coding: utf-8 -*-
"""Map trimmed VP1 alignment columns onto a real serotype A capsid structure.

PDB 2WZR is FMDV SAT1 and has no chain D. Its VP1 identity to A22/IRQ/24/64
is below the 70% gate, so the figure uses 4GH4 (FMDV A22 Iraq). VP1 in 4GH4
is chain A. Chain D of 4GH4 is VP4.
"""
import shutil
import urllib.request
from pathlib import Path

import pandas as pd
from Bio.Align import PairwiseAligner
from Bio.PDB import PDBParser, is_aa
from Bio.SeqUtils import seq1

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "step7_shap"
OUT = ROOT / "step8_shap"
FASTA = ROOT / "step3_alignment" / "aligned_vp1_protein_final.fasta"
CONFIRMED = [
    3, 40, 46, 83, 93, 95, 96, 101,
    140, 142, 143, 149, 164, 167,
    171, 173, 193, 197,
]
MODIFIED = {"MSE": "M", "SEC": "U", "PYL": "O"}
COPIED = [
    "shap_test_values.csv",
    "shap_global_importance.csv",
    "shap_bar_global.png",
    "shap_beeswarm.png",
    "shap_dependence_top5.png",
    "shap_waterfall_examples.png",
    "vp1_position_map.png",
    "serotype_comparison_table.csv",
    "shap_case_study_iran05.csv",
    "shap_case_study_iran05.png",
    "shap_interpretation_report.txt",
]


def read_fasta(path):
    records = []
    header = None
    chunks = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(chunks)))
            header = line[1:]
            chunks = []
        elif line:
            chunks.append(line.strip())
    if header is not None:
        records.append((header, "".join(chunks)))
    return records


def residue_letter(residue):
    try:
        return seq1(residue.resname)
    except KeyError:
        return MODIFIED.get(residue.resname)


def download(pdb_id):
    path = OUT / f"{pdb_id}.pdb"
    if not path.exists() or path.stat().st_size < 1000:
        urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdb_id}.pdb", path)
    return path


def title_lines(path):
    lines = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(("HEADER", "TITLE", "COMPND", "SOURCE")):
            lines.append(line.rstrip())
        elif line.startswith("ATOM"):
            break
    return lines


def full_chain(structure, chain_id):
    """Modeled residues plus REMARK 465 residues, in PDB number order."""
    modeled = {}
    for residue in structure[0][chain_id]:
        if not is_aa(residue, standard=False):
            continue
        letter = residue_letter(residue)
        if letter is None:
            continue
        _hetero, number, icode = residue.id
        modeled[(number, icode.strip())] = letter
    missing = {}
    for record in structure.header.get("missing_residues", []):
        if record.get("chain") != chain_id:
            continue
        model = record.get("model")
        if model not in (None, 0, 1):
            continue
        icode = record.get("insertion") or ""
        if icode in (" ", "?"):
            icode = ""
        try:
            letter = seq1(record["res_name"])
        except KeyError:
            continue
        missing[(int(record["ssseq"]), icode.strip())] = letter
    combined = dict(missing)
    combined.update(modeled)
    residues = []
    for number, icode in sorted(combined, key=lambda item: (item[0], item[1])):
        residues.append({
            "number": number,
            "icode": icode,
            "aa": combined[(number, icode)],
            "modeled": (number, icode) in modeled,
        })
    return residues


def identity(seq_a, seq_b):
    pairs = [(a, b) for a, b in zip(seq_a, seq_b) if a != "-" and b != "-"]
    matches = sum(a == b for a, b in pairs)
    return matches, len(pairs)


def align_chain(residues, ref_nogap, aligner):
    pdb_seq = "".join(item["aa"] for item in residues)
    alignment = aligner.align(pdb_seq, ref_nogap)[0]
    seq_a, seq_b = str(alignment[0]), str(alignment[1])
    if seq_a.replace("-", "") != pdb_seq or seq_b.replace("-", "") != ref_nogap:
        raise RuntimeError("pairwise alignment changed one of the sequences")
    matches, total = identity(seq_a, seq_b)
    return alignment, seq_a, seq_b, matches, total


def map_columns(ref_aligned, residues, seq_a, seq_b, chain_id, pdb_id):
    column_to_nogap = {}
    nogap_pos = 0
    for column, amino in enumerate(ref_aligned, start=1):
        if amino not in "-X":
            column_to_nogap[column] = nogap_pos
            nogap_pos += 1
    nogap_to_pdb = {}
    pdb_index = 0
    nogap_index = 0
    for pdb_aa, ref_aa in zip(seq_a, seq_b):
        if pdb_aa != "-" and ref_aa != "-":
            item = residues[pdb_index]
            nogap_to_pdb[nogap_index] = item
            pdb_index += 1
            nogap_index += 1
        elif pdb_aa == "-" and ref_aa != "-":
            nogap_index += 1
        elif pdb_aa != "-" and ref_aa == "-":
            pdb_index += 1
        else:
            raise RuntimeError("pairwise alignment contains a double gap")
    rows = []
    for column in range(1, 214):
        amino = ref_aligned[column - 1]
        gapped = amino in "-X"
        item = None if gapped else nogap_to_pdb.get(column_to_nogap[column])
        rows.append({
            "alignment_col": column,
            "aa_in_ref": amino,
            "is_gap": gapped,
            "pdb_id": pdb_id,
            "pdb_chain": chain_id if item else "",
            "pdb_resnum": None if item is None else item["number"],
            "pdb_icode": "" if item is None else item["icode"],
            "pdb_aa": None if item is None else item["aa"],
            "modeled": False if item is None else item["modeled"],
            "aa_matches_pdb": bool(item and amino == item["aa"]),
            "is_confirmed": column in CONFIRMED,
        })
    return pd.DataFrame(rows)


def write_pml(mapping, shap_of, pdb_id, chain_id, matches, total):
    confirmed = mapping[mapping["is_confirmed"]].copy()

    def category(column):
        value = shap_of.get(f"site_{int(column):03d}_diff", 0)
        if value >= 0.04:
            return "high"
        if value >= 0.02:
            return "medium"
        return "low"

    drawable = confirmed[confirmed["modeled"]].copy()
    drawable["shap_cat"] = drawable["alignment_col"].map(category)
    disordered = confirmed[confirmed["pdb_resnum"].notna() & ~confirmed["modeled"]]
    absent = confirmed[confirmed["pdb_resnum"].isna()]
    gh = mapping[(mapping["alignment_col"].between(132, 161)) & mapping["modeled"]]
    groups = {
        "high": drawable.loc[drawable["shap_cat"] == "high", "pdb_resnum"].astype(int).tolist(),
        "medium": drawable.loc[drawable["shap_cat"] == "medium", "pdb_resnum"].astype(int).tolist(),
        "low": drawable.loc[drawable["shap_cat"] == "low", "pdb_resnum"].astype(int).tolist(),
    }

    def resi(numbers):
        return "+".join(str(number) for number in sorted(set(int(number) for number in numbers)))

    lines = [
        "# PyMOL script — FMDV serotype A VP1 positions",
        "# 2WZR was not used: it is serotype SAT1, it has no chain D,",
        "# and its VP1 identity to A22/IRQ/24/64 is 56.1%.",
        f"# Structure: {pdb_id}, FMDV A22 Iraq. VP1 is chain {chain_id}.",
        f"# Reference sequence: A22/IRQ/24/64. Identity {matches}/{total} = {matches / total * 100:.1f}%.",
        "# Residue numbers are the PDB author numbers.",
        "",
        f"fetch {pdb_id}",
        "remove solvent",
        "hide everything",
        f"show cartoon, chain {chain_id}",
        f"color gray80, chain {chain_id}",
        "",
    ]
    if len(gh):
        lines += [
            f"select gh_loop, chain {chain_id} and resi {int(gh['pdb_resnum'].min())}-{int(gh['pdb_resnum'].max())}",
            "color salmon, gh_loop",
            "",
        ]
    labels = {
        "high": "High importance (mean |SHAP| >= 0.04), coordinates present",
        "medium": "Medium importance (0.02 <= mean |SHAP| < 0.04), coordinates present",
        "low": "Low importance (mean |SHAP| < 0.02), coordinates present",
    }
    colors = {"high": "red", "medium": "orange", "low": "yellow"}
    for key in ("high", "medium", "low"):
        if groups[key]:
            lines += [
                f"# {labels[key]}",
                f"select {key}_shap, chain {chain_id} and resi {resi(groups[key])}",
                f"show sticks, {key}_shap",
                f"color {colors[key]}, {key}_shap",
                "",
            ]
    if len(disordered):
        lines.append("# These confirmed positions map to PDB numbers but have no coordinates (disordered loop):")
        for _, row in disordered.iterrows():
            lines.append(
                f"# alignment {int(row['alignment_col']):03d} {row['aa_in_ref']} -> "
                f"{pdb_id} chain {chain_id} resi {int(row['pdb_resnum'])} "
                f"({category(row['alignment_col'])} SHAP)"
            )
        lines.append("")
    if len(absent):
        lines.append("# Confirmed alignment columns with no PDB residue in the reference: "
                     + ",".join(str(int(value)) for value in absent["alignment_col"]))
    lines += [
        f"zoom chain {chain_id}",
        "set ray_shadows, 0",
        "ray 1200, 900",
        "png vp1_structure_shap_positions.png, dpi=300",
    ]
    (OUT / "color_vp1_positions_pdb.pml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return groups, disordered, absent


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name in COPIED:
        shutil.copy2(SRC / name, OUT / name)

    records = read_fasta(FASTA)
    ref_name, ref_aligned = next(item for item in records if item[0].split("|")[0] == "A22/IRQ/24/64")
    if len(ref_aligned) != 213:
        raise SystemExit(f"reference alignment length is {len(ref_aligned)}, expected 213")
    ref_nogap = ref_aligned.replace("-", "").replace("X", "")
    print(f"reference {ref_name.split('|')[0]} ungapped length {len(ref_nogap)}", flush=True)

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -5
    aligner.extend_gap_score = -0.5

    pdb_id = "4GH4"
    path = download(pdb_id)
    print("\n".join(title_lines(path)[:24]), flush=True)
    structure = PDBParser(QUIET=True).get_structure(pdb_id, str(path))
    ranked = []
    for chain in structure[0]:
        residues = full_chain(structure, chain.id)
        if len(residues) < 150:
            print(f"chain {chain.id}: {len(residues)} residues, skipped", flush=True)
            continue
        _alignment, seq_a, seq_b, matches, total = align_chain(residues, ref_nogap, aligner)
        modeled = sum(item["modeled"] for item in residues)
        print(f"chain {chain.id}: sequence {len(residues)} ({modeled} modeled), identity {matches}/{total} = {matches / total * 100:.1f}%", flush=True)
        ranked.append((matches / total, matches, total, chain.id, residues, seq_a, seq_b))
    ranked.sort(reverse=True, key=lambda item: item[0])
    fraction, matches, total, chain_id, residues, seq_a, seq_b = ranked[0]
    if fraction < 0.70:
        raise SystemExit(f"{pdb_id} best chain {chain_id} identity {fraction * 100:.1f}% is below 70%")
    print(f"\nselected {pdb_id} chain {chain_id}: {matches}/{total} = {fraction * 100:.1f}%", flush=True)
    print(f"PDB: {seq_a[:90]}", flush=True)
    print(f"A22: {seq_b[:90]}", flush=True)

    mapping = map_columns(ref_aligned, residues, seq_a, seq_b, chain_id, pdb_id)
    mapping.to_csv(OUT / "alignment_to_pdb_mapping.csv", index=False, encoding="utf-8-sig")
    confirmed = mapping[mapping["is_confirmed"]]
    print("\n18 confirmed positions:", flush=True)
    print(confirmed[["alignment_col", "aa_in_ref", "pdb_resnum", "pdb_aa", "modeled", "aa_matches_pdb"]].to_string(index=False), flush=True)
    print("\ncolumns 140-150:", flush=True)
    window = mapping[mapping["alignment_col"].between(140, 150)]
    print(window[["alignment_col", "aa_in_ref", "pdb_resnum", "pdb_aa", "modeled"]].to_string(index=False), flush=True)

    importance = pd.read_csv(OUT / "shap_global_importance.csv", encoding="utf-8-sig")
    shap_of = dict(zip(importance["feature"], importance["mean_abs_shap"]))
    groups, disordered, absent = write_pml(mapping, shap_of, pdb_id, chain_id, matches, total)
    print(f"\nhigh {sorted(groups['high'])}", flush=True)
    print(f"medium {sorted(groups['medium'])}", flush=True)
    print(f"low {sorted(groups['low'])}", flush=True)
    print(f"disordered {[int(v) for v in disordered['alignment_col']]}", flush=True)
    print(f"absent {[int(v) for v in absent['alignment_col']]}", flush=True)

    mapped = confirmed[confirmed["pdb_resnum"].notna()]
    required = COPIED + ["alignment_to_pdb_mapping.csv", "color_vp1_positions_pdb.pml"]
    checks = {name: (OUT / name).exists() for name in required}
    checks["mapping has 213 rows"] = len(mapping) == 213
    checks["column 149 has a PDB number"] = bool(mapping.loc[mapping["alignment_col"] == 149, "pdb_resnum"].notna().any())
    checks["at least 15 of 18 confirmed positions map"] = len(mapped) >= 15
    checks["PML uses PDB residue numbers"] = "fetch 4GH4" in (OUT / "color_vp1_positions_pdb.pml").read_text(encoding="utf-8")
    print("\nCHECKS", flush=True)
    for name, ok in checks.items():
        print(f"  {'OK' if ok else 'FAIL'} {name}", flush=True)
    print(f"READY {all(checks.values())}  mapped {len(mapped)}/18", flush=True)


if __name__ == "__main__":
    main()
