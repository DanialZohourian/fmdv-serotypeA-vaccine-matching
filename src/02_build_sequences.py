# -*- coding: utf-8 -*-
"""Step 2: download FMDV serotype A records and write QC'd VP1 FASTA files.

Field-virus accessions come from dataset/master_pairs.csv.
Vaccine-serum accessions were checked against GenBank titles. Several
accessions suggested in the protocol belong to other organisms and are
not used. VP1 is taken from the GenBank feature table. A fixed genome
coordinate is not used, because VP1 is 633-639 nt, not a 744 nt window.
"""

import csv
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from Bio import Entrez, SeqIO
from Bio.Seq import Seq

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "dataset" / "master_pairs.csv"
OUT = ROOT / "step2_sequences"

Entrez.email = "research@example.com"
Entrez.tool = "fmdv_step2"

# name, accession, lineage, country, year, mapping note
# Only records whose definition line is the vaccine strain, or the
# public isolate customarily used for that WRL serum name.
SERA = [
    ("A22/IRQ/24/64", "AY593763.1", "A22", "Iraq", "1964",
     "Complete genome a22iraq64 iso86. AJ539174.1 is a bacterial 16S rRNA gene and was rejected."),
    ("A/IRN/1996", "EF208771.1", "A/ASIA/Iran-96", "Iran", "1996",
     "VP1 of A/IRN/1/96, the public representative of the A Iran 96 vaccine. AJ251473.1 is SAT2 KEN/3/57 and was rejected."),
    ("A/IRN/1999", "EF208772.1", "A/ASIA/Iran-99", "Iran", "1999",
     "No record is named A/IRN/1999. Used A/IRN/22/99 VP1, the serotype A Iran 1999 sequence found in GenBank."),
    ("A/IRN/1987", "EF208770.1", "A/ASIA/Iran-87", "Iran", "1987",
     "VP1 of A/IRN/2/87, the public representative of the A Iran 87 vaccine."),
    ("A/SAU/1995", "EU553874.1", "A/ASIA/Saudi-95", "Saudi Arabia", "1995",
     "VP1 of A/SAU/16/95. No record is named exactly A/SAU/1995; this is the closest 1995 Saudi serotype A VP1."),
    ("A Sau 41/91 SI94", "FJ755085.1", "A/ASIA/Saudi-91", "Saudi Arabia", "1991",
     "VP1 of A/SAU/41/91."),
    ("A/TUR/2006", "FJ755116.1", "A/ASIA/Iran-05", "Turkey", "2006",
     "VP1 of A/TUR/20/2006, matching the WRL label A/TUR/20/06. EU257510.1 is a plant carlavirus and was rejected."),
    ("A/TUR/4/2006", "FJ755103.1", "A/ASIA/Iran-05", "Turkey", "2006",
     "VP1 of A/TUR/4/2006."),
    ("A/Iran-05", "EF208769.1", "A/ASIA/Iran-05", "Iran", "2005",
     "VP1 of prototype A/IRN/1/2005. DQ533483 is serotype Asia 1 strain ZB/CHA/58 and was rejected."),
    ("A/ASIA/G-VII/2015", "KY982283.1", "A/ASIA/G-VII", "Iran", "2015",
     "VP1 of IRN/12/2015, a public A/ASIA/G-VII 2015 reference. KX894955.1 is a fungal ITS sequence and was rejected."),
    ("A/ERI/1998", "KX258045.1", "A/AFRICA/Eritrea-98", "Eritrea", "1998",
     "Same deposit as A/ERI/3/1998. GenBank names this vaccine ERI/3/98; no separate A/ERI/1998 VP1 exists."),
    ("A/ERI/3/1998", "KX258045.1", "A/AFRICA/Eritrea-98", "Eritrea", "1998",
     "VP1 of ERI/3/98."),
    ("A/KEN/35/1980", "KJ440846.1", "A/AFRICA/Kenya-80", "Kenya", "1980",
     "VP1 of K35/80."),
    ("A/IND/17/1982", "HM854024.1", "A/ASIA/India-82", "India", "1982",
     "Complete genome IND17/82. VP1 taken from the feature table."),
    ("A24 Cruzeiro", "JQ082960.1", "A24", "Brazil", "1955",
     "VP1 of A24/Cruzeiro/Bra/55. AJ320488.1 is O1 Campos and was rejected."),
    ("A15 Tai 1/60", "KF112918.1", "A15", "Thailand", "1960",
     "Capsid of A15/Bangkok/TAI/60. VP1 taken from the feature table."),
]

# WRL serum names that still have no defensible public VP1.
SERUM_GAPS = [
    ("A/IRN/2001", "No nucleotide record titled A/IRN/2001 or A/IRN/1/2001 was found."),
    ("A/MAY/1997", "No nucleotide record titled A/MAY/1997 or A/MAY/97 was found."),
    ("A/5925", "The token 5925 matches patents and unrelated genomes, not an FMDV A5925 VP1."),
    ("A24 UV64", "No FMDV record containing UV64 was found."),
    ("A24 TE48", "No FMDV record containing TE48 was found."),
    ("A15", "Unqualified WRL name A15. A15/Bangkok/TAI/60 was kept for A15 Tai 1/60 and was not copied onto this shorter name."),
]


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def mode_value(values):
    filled = [v for v in values if v]
    if not filled:
        return ""
    return Counter(filled).most_common(1)[0][0]


def load_master():
    with MASTER.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    viruses = defaultdict(lambda: {"lineage": [], "country": [], "year": [], "n": 0})
    sera_n = Counter()
    for row in rows:
        viruses[row["virus_name"]]["lineage"].append(row["lineage"])
        viruses[row["virus_name"]]["country"].append(row["country"])
        viruses[row["virus_name"]]["year"].append(row["year_isolated"])
        viruses[row["virus_name"]]["n"] += 1
        viruses[row["virus_name"]]["accession"] = row["virus_accession"]
        sera_n[row["serum_name"]] += 1
    catalog = []
    for name, meta in sorted(viruses.items()):
        catalog.append({
            "name": name,
            "accession": meta["accession"],
            "lineage": mode_value(meta["lineage"]),
            "country": mode_value(meta["country"]),
            "year": mode_value(meta["year"]),
            "type": "virus",
            "pairs_count": meta["n"],
            "mapping_note": "",
        })
    return catalog, sera_n


def download_genbank(accession, dest):
    if dest.exists() and dest.stat().st_size > 200:
        return
    last = None
    for attempt in range(4):
        try:
            handle = Entrez.efetch(db="nucleotide", id=accession, rettype="gb", retmode="text")
            text = handle.read()
            handle.close()
            if isinstance(text, bytes):
                text = text.decode("utf-8", "replace")
            if "LOCUS" not in text[:500]:
                raise RuntimeError("response was not a GenBank record")
            dest.write_text(text, encoding="utf-8")
            time.sleep(0.4)
            return
        except Exception as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"download failed for {accession}: {last}")


def feature_text(feature):
    parts = []
    for key in ("product", "gene", "note"):
        parts.extend(feature.qualifiers.get(key, []))
    return " ".join(parts).lower()


def is_vp1_text(text):
    if "vp1" in text:
        return True
    return re.search(r"\b1d\b", text) is not None


def vp1_nucleotide_from_open_frame(nucleotide):
    """Cut VP1 from a capsid or polyprotein CDS using the RGD site.

    In serotype A the arginine of RGD is residue 145 of VP1. On a complete
    polyprotein the VP1/2A junction is the NFDLL motif. A capsid-only record
    ends at the VP1 C-terminus, so the cut runs from RGD-144 to the end.
    """
    nucleotide = nucleotide.upper().replace("U", "T")
    nucleotide = nucleotide[: len(nucleotide) - (len(nucleotide) % 3)]
    protein = str(Seq(nucleotide).translate(to_stop=False))
    if protein.endswith("*"):
        protein = protein[:-1]
        nucleotide = nucleotide[: len(protein) * 3]
    starts = []
    cursor = 0
    while True:
        site = protein.find("RGD", cursor)
        if site < 0:
            break
        if site >= 144:
            starts.append(site)
        cursor = site + 1
    if not starts:
        return "", ""
    for site in starts:
        junction = protein.find("NFDLL", site)
        if 0 < junction - (site - 144) <= 230:
            start = site - 144
            return nucleotide[start * 3:junction * 3], "rgd_to_2a_junction"
    if len(protein) < 1000:
        site = starts[-1]
        if site > len(protein) - 250:
            start = site - 144
            return nucleotide[start * 3:], "rgd_capsid_to_end"
    return "", ""


def extract_vp1(record):
    short = []
    long_cds = []
    for feature in record.features:
        if feature.type not in {"CDS", "mat_peptide"}:
            continue
        text = feature_text(feature)
        seq = str(feature.extract(record.seq)).upper().replace("U", "T")
        if is_vp1_text(text) and 600 <= len(seq) <= 700:
            short.append((0 if feature.type == "CDS" else 1, abs(len(seq) - 639), seq))
        elif len(seq) >= 1500 and (
            "capsid" in text or "polyprotein" in text or "p1" in text or is_vp1_text(text)
        ):
            long_cds.append(seq)
    if short:
        short.sort()
        return short[0][2], "annotation"
    whole = str(record.seq).upper().replace("U", "T")
    if 600 <= len(whole) <= 700:
        return whole, "record_is_vp1"
    for seq in sorted(long_cds, key=len):
        cut, source = vp1_nucleotide_from_open_frame(seq)
        if cut:
            return cut, source
    if len(whole) >= 1500:
        cut, source = vp1_nucleotide_from_open_frame(whole)
        if cut:
            return cut, source
    return "", "not_found"


def translate_best(nucleotide):
    """Translate the frame that yields a VP1-length protein without an internal stop."""
    nucleotide = nucleotide.upper().replace("U", "T")
    best = None
    for frame in (0, 1, 2):
        chunk = nucleotide[frame:]
        chunk = chunk[: len(chunk) - (len(chunk) % 3)]
        if len(chunk) < 600:
            continue
        protein = str(Seq(chunk).translate(to_stop=False)).rstrip("*")
        internal = "*" in protein
        score = (
            0 if not internal else 1,
            0 if 208 <= len(protein) <= 220 else 1,
            abs(len(protein) - 213),
        )
        item = (score, frame, protein, chunk)
        if best is None or item[0] < best[0]:
            best = item
    if best is None:
        return "", "empty"
    _, frame, protein, _ = best
    note = "" if frame == 0 else f"translation_frame_{frame}"
    return protein, note


def qc_check(nucleotide, protein, frame_note):
    issues = []
    notes = []
    if frame_note:
        notes.append(frame_note)
    if not nucleotide:
        return "FLAG", "vp1_not_found", notes
    n_count = sum(1 for base in nucleotide if base not in "ACGT")
    n_percent = n_count / len(nucleotide)
    if not (630 <= len(nucleotide) <= 660):
        if 607 <= len(nucleotide) <= 671:
            notes.append(f"nuc_length_near_vp1:{len(nucleotide)}")
        else:
            issues.append(f"nuc_length_unusual:{len(nucleotide)}")
    if n_percent > 0.05:
        issues.append(f"high_ambiguous:{n_percent:.1%}")
    if not protein:
        issues.append("no_protein")
    else:
        if "*" in protein:
            issues.append("internal_stop_codon")
        if not (208 <= len(protein) <= 220):
            if 200 <= len(protein) <= 226 and "internal_stop_codon" not in issues:
                notes.append(f"protein_length_near_vp1:{len(protein)}")
            else:
                issues.append(f"protein_length_unusual:{len(protein)}")
        if not protein.startswith("M"):
            # VP1 is cleaved from the polyprotein. A threonine start is normal.
            notes.append("vp1_has_no_initiator_methionine")
        if "RGD" not in protein:
            notes.append("rgd_motif_not_found")
    status = "PASS" if not issues else "FLAG"
    return status, ";".join(issues), notes


def header_of(item):
    return "|".join([
        item["name"],
        item["accession"],
        item["lineage"],
        item["country"],
        item["year"],
        item["type"],
    ])


def write_fasta(path, header, sequence):
    lines = [f">{header}"]
    for i in range(0, len(sequence), 60):
        lines.append(sequence[i:i + 60])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    virus_catalog, sera_counts = load_master()
    for folder in (
        OUT / "01_raw_downloads" / "viruses",
        OUT / "01_raw_downloads" / "sera",
        OUT / "02_vp1_extracted" / "viruses",
        OUT / "02_vp1_extracted" / "sera",
        OUT / "03_vp1_protein" / "viruses",
        OUT / "03_vp1_protein" / "sera",
        OUT / "04_qc",
        OUT / "05_final",
    ):
        folder.mkdir(parents=True, exist_ok=True)

    serum_catalog = []
    for name, acc, lineage, country, year, note in SERA:
        serum_catalog.append({
            "name": name,
            "accession": acc,
            "lineage": lineage,
            "country": country,
            "year": year,
            "type": "serum",
            "pairs_count": sera_counts[name],
            "mapping_note": note,
        })

    print(f"viruses {len(virus_catalog)} sera_with_accession {len(serum_catalog)}", flush=True)
    for item in virus_catalog:
        dest = OUT / "01_raw_downloads" / "viruses" / f"{item['accession']}.gb"
        download_genbank(item["accession"], dest)
        print(f"virus {item['accession']} {item['name']}", flush=True)
    seen_serum_acc = set()
    for item in serum_catalog:
        if item["accession"] in seen_serum_acc:
            continue
        seen_serum_acc.add(item["accession"])
        dest = OUT / "01_raw_downloads" / "sera" / f"{item['accession']}.gb"
        download_genbank(item["accession"], dest)
        print(f"serum {item['accession']} {item['name']}", flush=True)

    qc_rows = []
    flagged = []
    passed = []
    for item in virus_catalog + serum_catalog:
        folder = "viruses" if item["type"] == "virus" else "sera"
        gb_path = OUT / "01_raw_downloads" / folder / f"{item['accession']}.gb"
        record = SeqIO.read(str(gb_path), "genbank")
        nucleotide, source = extract_vp1(record)
        protein, frame_note = translate_best(nucleotide) if nucleotide else ("", "")
        status, issues, notes = qc_check(nucleotide, protein, frame_note)
        if item["mapping_note"]:
            notes.append(item["mapping_note"])
        n_percent = 0.0
        if nucleotide:
            n_percent = sum(base not in "ACGT" for base in nucleotide) / len(nucleotide)
        row = {
            "sequence_id": item["name"],
            "type": item["type"],
            "accession": item["accession"],
            "nuc_length": len(nucleotide),
            "protein_length": len(protein),
            "ambiguous_percent": f"{n_percent:.1%}",
            "vp1_source": source,
            "status": status,
            "issues": issues,
            "notes": "; ".join(notes),
        }
        qc_rows.append(row)
        item.update({"nucleotide": nucleotide, "protein": protein, "vp1_source": source, "qc_status": status})
        if status == "PASS":
            passed.append(item)
            stem = safe_name(item["name"])
            write_fasta(
                OUT / "02_vp1_extracted" / folder / f"{stem}_VP1.fasta",
                header_of(item), nucleotide,
            )
            write_fasta(
                OUT / "03_vp1_protein" / folder / f"{stem}_VP1_protein.fasta",
                header_of(item), protein,
            )
        else:
            flagged.append(f"{item['name']}\t{item['type']}\tFLAG\t{issues}\t{row['notes']}")
        print(f"QC {status} {item['type']} {item['name']} nuc={len(nucleotide)} aa={len(protein)} {source}", flush=True)

    for name, reason in SERUM_GAPS:
        qc_rows.append({
            "sequence_id": name,
            "type": "serum",
            "accession": "",
            "nuc_length": "",
            "protein_length": "",
            "ambiguous_percent": "",
            "vp1_source": "",
            "status": "NOT_FOUND",
            "issues": reason,
            "notes": "",
        })
        flagged.append(f"{name}\tserum\tNOT_FOUND\t{reason}")

    qc_fields = [
        "sequence_id", "type", "accession", "nuc_length", "protein_length",
        "ambiguous_percent", "vp1_source", "status", "issues", "notes",
    ]
    with (OUT / "04_qc" / "qc_report.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=qc_fields)
        writer.writeheader()
        writer.writerows(qc_rows)
    (OUT / "04_qc" / "flagged.txt").write_text(
        "\n".join(flagged) + ("\n" if flagged else ""), encoding="utf-8"
    )

    passed_viruses = [item for item in passed if item["type"] == "virus"]
    passed_sera = [item for item in passed if item["type"] == "serum"]
    nuc_chunks = []
    aa_chunks = []
    for item in passed_viruses + passed_sera:
        header = header_of(item)
        nuc_chunks.append(f">{header}\n" + wrap(item["nucleotide"]))
        aa_chunks.append(f">{header}\n" + wrap(item["protein"]))
    (OUT / "05_final" / "vp1_all_nucleotide.fasta").write_text("\n".join(nuc_chunks) + "\n", encoding="utf-8")
    (OUT / "05_final" / "vp1_all_protein.fasta").write_text("\n".join(aa_chunks) + "\n", encoding="utf-8")

    registry = []
    for index, item in enumerate(passed_viruses + passed_sera, 1):
        registry.append({
            "seq_id": index,
            "virus_name": item["name"],
            "accession": item["accession"],
            "type": item["type"],
            "nuc_length": len(item["nucleotide"]),
            "protein_length": len(item["protein"]),
            "vp1_source": item["vp1_source"],
            "qc_status": item["qc_status"],
            "pairs_count": item["pairs_count"],
            "in_master_pairs": "YES" if item["pairs_count"] else "NO",
            "mapping_note": item["mapping_note"],
        })
    next_id = len(registry) + 1
    present = {item["name"] for item in passed}
    for item in virus_catalog + serum_catalog:
        if item["name"] in present:
            continue
        registry.append({
            "seq_id": next_id,
            "virus_name": item["name"],
            "accession": item["accession"],
            "type": item["type"],
            "nuc_length": "",
            "protein_length": "",
            "vp1_source": item.get("vp1_source", ""),
            "qc_status": item.get("qc_status", "FLAG"),
            "pairs_count": item["pairs_count"],
            "in_master_pairs": "YES",
            "mapping_note": item["mapping_note"],
        })
        next_id += 1
    for name, reason in SERUM_GAPS:
        registry.append({
            "seq_id": next_id,
            "virus_name": name,
            "accession": "",
            "type": "serum",
            "nuc_length": "",
            "protein_length": "",
            "vp1_source": "",
            "qc_status": "NOT_FOUND",
            "pairs_count": sera_counts[name],
            "in_master_pairs": "YES",
            "mapping_note": reason,
        })
        next_id += 1

    reg_fields = [
        "seq_id", "virus_name", "accession", "type", "nuc_length", "protein_length",
        "vp1_source", "qc_status", "pairs_count", "in_master_pairs", "mapping_note",
    ]
    with (OUT / "05_final" / "sequence_registry.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=reg_fields)
        writer.writeheader()
        writer.writerows(registry)

    protein_names = set()
    bad_length = {}
    # SeqIO splits a FASTA id at the first space, which would break
    # serum names such as "A24 Cruzeiro". Read the raw header instead.
    header = None
    length = 0
    protein_path = OUT / "05_final" / "vp1_all_protein.fasta"
    for line in protein_path.read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            if header is not None and not (208 <= length <= 220):
                bad_length[header.split("|")[0]] = length
            header = line[1:]
            protein_names.add(header.split("|")[0])
            length = 0
        else:
            length += len(line.strip())
    if header is not None and not (208 <= length <= 220):
        bad_length[header.split("|")[0]] = length
    master_viruses = {item["name"] for item in virus_catalog}
    master_sera = set(sera_counts)
    missing_viruses = sorted(master_viruses - protein_names)
    missing_sera = sorted(master_sera - protein_names)
    lines = [
        "Step 2 validation",
        f"total_sequences: {len(protein_names)}",
        f"virus_sequences: {sum(1 for name in protein_names if name in master_viruses)}",
        f"serum_sequences: {sum(1 for name in protein_names if name in master_sera)}",
        f"missing_viruses ({len(missing_viruses)}): {', '.join(missing_viruses) or 'none'}",
        f"missing_sera ({len(missing_sera)}): {', '.join(missing_sera) or 'none'}",
        f"bad_length ({len(bad_length)}): {bad_length or 'none'}",
        "",
        "Green status needs no missing field virus, no missing serum, and no protein outside 208-220 aa.",
        (
            "Green: every master_pairs virus and serum has a VP1 protein of 208-220 aa."
            if not missing_viruses and not missing_sera and not bad_length
            else "Not green: at least one master_pairs name has no accepted VP1."
        ),
        "VP1 proteins do not start with methionine; that is the capsid cleavage site, not a failed translation.",
        "A fixed VP1 window of genome positions 2541-3285 was not used. That window is 744 nt.",
    ]
    text = "\n".join(lines) + "\n"
    (OUT / "05_final" / "validation.txt").write_text(text, encoding="utf-8")
    print(text, flush=True)


def wrap(sequence):
    return "\n".join(sequence[i:i + 60] for i in range(0, len(sequence), 60))


if __name__ == "__main__":
    main()
