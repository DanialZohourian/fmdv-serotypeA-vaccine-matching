# -*- coding: utf-8 -*-
"""Build the standardized FMDV serotype A r1 master table.

One row is one serum-virus pair. Inclusion (all required):
  - WOAH-style 2D-VNT (not LPBE / ELISA)
  - bovine vaccinate serum
  - an exact numeric r1 (censored bounds are kept in the exclusion log)
  - a VP1 or P1 sequence stored from GenBank
"""

import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset"

# Spelling variants of the same vaccine serum. Strain identities that
# differ by isolate number are not merged.
SERUM_ALIASES = {
    "A Malaysia 97": "A/MAY/1997",
    "A IRN 01": "A/IRN/2001",
    "A 24 UV64": "A24 UV64",
    "A K35/80": "A/KEN/35/1980",
    "A Sau41/91 SI94": "A Sau 41/91 SI94",
    "A Eri": "A/ERI/1998",
}

HOST_MAP = {
    "bovine": "cattle",
    "cattle": "cattle",
    "ovine": "sheep",
    "buffalo": "buffalo",
    "nk": "unknown",
}


def clean(value):
    return (value or "").strip()


def norm_lineage(raw):
    token = clean(raw)
    key = token.lower().replace("_", "-").replace(" ", "")
    if key in {"", "nd", "na", "unknown"}:
        return ""
    if key in {"iran-05", "a-iran-2005", "a/asia/iran-05", "iran05"}:
        return "A/ASIA/Iran-05"
    if key in {"iran-96", "a/asia/iran-96"}:
        return "A/ASIA/Iran-96"
    if key == "a22":
        return "A22"
    if key in {"i", "ii", "iv", "vii"}:
        return f"genotype {token}"
    return token


def norm_sublineage(raw):
    token = clean(raw)
    if token.lower() in {"", "nd", "na", "unknown"}:
        return ""
    return token


def norm_host(raw):
    token = clean(raw)
    if not token:
        return ""
    return HOST_MAP.get(token.lower(), token.lower())


def norm_serum(raw):
    token = clean(raw)
    return SERUM_ALIASES.get(token, token)


def source_label(publication):
    pub = clean(publication)
    if pub.startswith("Singanallur"):
        return "Singanallur et al. 2022, Viruses 14:97, Table 1"
    if pub.startswith("Azimi"):
        return "Azimi et al. 2020, Arch Razi Inst 75:349-357"
    return pub


def data_format_of(location):
    # Every current numeric source is an exact published number.
    # "coordinate" is PDF text-layer extraction, not figure digitization.
    loc = clean(location)
    if loc in {"pdf_table", "literature_table", "coordinate"}:
        return "table"
    return ""


def format_note(location):
    # Ruled tables and paper tables need no per-row note: data_format is "table".
    # Word-position reads are still exact published digits, so they stay "table",
    # but the note records that they were not copied from a ruled grid.
    if clean(location) == "coordinate":
        return "exact r1 from the PDF text layer (word position), not from a figure"
    return ""


def choose_sequence(records):
    usable = [r for r in records if clean(r.get("sequence_type")) in {"VP1", "P1"}]
    if not usable:
        return None

    def rank(rec):
        kind = clean(rec.get("sequence_type"))
        score = int(rec.get("match_score") or 0)
        length = int(rec.get("sequence_length") or 0)
        return (0 if kind == "VP1" else 1, -score, -length)

    return sorted(usable, key=rank)[0]


def load_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fieldnames):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    pairs = load_csv(DATA / "pairs.csv")
    viruses = load_csv(DATA / "viruses.csv")
    sequences = load_csv(DATA / "sequences.csv")

    virus_meta = {clean(v["field_virus"]): v for v in viruses}
    seq_by_virus = defaultdict(list)
    for rec in sequences:
        seq_by_virus[clean(rec["virus"])].append(rec)

    prepared = []
    for row in pairs:
        virus = clean(row["field_virus"])
        serum_raw = clean(row["vaccine_strain"])
        serum = norm_serum(serum_raw)
        meta = virus_meta.get(virus, {})
        seq = choose_sequence(seq_by_virus.get(virus, []))

        lineage_pair = clean(row.get("lineage"))
        lineage_meta = clean(meta.get("lineage"))
        lineage_source = lineage_meta or lineage_pair
        lineage = norm_lineage(lineage_source)

        sub_pair = clean(row.get("sublineage"))
        sub_meta = clean(meta.get("sublineage"))
        sub_source = sub_meta or sub_pair
        sub_lineage = norm_sublineage(sub_source)

        country = clean(row.get("country")) or clean(meta.get("country"))
        year = clean(str(row.get("collection_year") or "")) or clean(str(meta.get("collection_year") or ""))
        host = norm_host(meta.get("host"))

        accession = ""
        sequence_type = ""
        if seq:
            accession = clean(seq.get("genbank_accession"))
            sequence_type = clean(seq.get("sequence_type"))

        operator = clean(row.get("r1_operator"))
        r1_text = clean(row.get("r1"))
        assay = clean(row.get("assay_type"))
        location = clean(row.get("data_location"))

        reasons = []
        if assay != "VNT":
            reasons.append("not_2d_vnt")
        if operator != "=" or not r1_text:
            if not r1_text:
                reasons.append("r1_missing")
            else:
                reasons.append("r1_censored")
        if not accession or sequence_type not in {"VP1", "P1"}:
            reasons.append("no_vp1_or_p1_sequence")

        notes = []
        if serum != serum_raw and serum_raw:
            notes.append(f"serum name standardized from {serum_raw}")
        batch = clean(row.get("vaccine_raw"))
        if batch and batch != serum and batch != serum_raw:
            notes.append(f"serum label in source: {batch}")
        if lineage_source and lineage and lineage_source != lineage:
            notes.append(f"lineage standardized from {lineage_source}")
        if lineage_pair and lineage_meta and norm_lineage(lineage_pair) != norm_lineage(lineage_meta):
            notes.append(
                f"lineage labels differ: pair={lineage_pair}; catalogue={lineage_meta}"
            )
        if sequence_type:
            notes.append(f"sequence_type={sequence_type}")
        mapping = clean(seq.get("mapping_note")) if seq else ""
        if mapping:
            notes.append(mapping)
        loc_note = format_note(location)
        if loc_note:
            notes.append(loc_note)

        r1_value = ""
        r1_binary = ""
        if "r1_missing" not in reasons and "r1_censored" not in reasons:
            r1_value = r1_text
            r1_binary = "1" if float(r1_text) >= 0.3 else "0"

        prepared.append({
            "virus_name": virus,
            "virus_accession": accession,
            "serum_name": serum,
            "serum_raw": serum_raw,
            "r1_value": r1_value,
            "r1_text": r1_text,
            "r1_operator": operator,
            "r1_binary": r1_binary,
            "lineage": lineage,
            "sub_lineage": sub_lineage,
            "country": country,
            "year_isolated": year,
            "host_species": host,
            "VNT_method": "2D-VNT" if assay == "VNT" else assay,
            "data_source": source_label(row.get("publication")),
            "data_format": data_format_of(location),
            "notes": notes,
            "exclusion_reasons": reasons,
            "publication_file": clean(row.get("publication")),
            "source_label": batch or serum_raw or serum,
        })

    # Same virus + same serum + same report + same exact r1 is one pair.
    deduped = []
    seen = set()
    for rec in prepared:
        key = (
            rec["virus_name"],
            rec["serum_name"],
            rec["publication_file"],
            rec["r1_operator"],
            rec["r1_text"],
            rec["serum_raw"],
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(rec)

    def exact_r1(rec):
        return bool(rec["r1_value"]) and "r1_missing" not in rec["exclusion_reasons"] and (
            "r1_censored" not in rec["exclusion_reasons"]
        )

    # Contradictions are compared inside 2D-VNT only. An ELISA/LPBE number
    # for the same names is a different assay, not a second VNT result.
    vnt_by_pair = defaultdict(list)
    other_by_pair = defaultdict(list)
    for rec in deduped:
        if not exact_r1(rec):
            continue
        key = (rec["virus_name"], rec["serum_name"])
        if "not_2d_vnt" in rec["exclusion_reasons"]:
            other_by_pair[key].append(rec)
        else:
            vnt_by_pair[key].append(rec)

    contradiction_rows = []
    for (virus, serum), group in sorted(vnt_by_pair.items()):
        values = sorted({float(r["r1_value"]) for r in group})
        sources = sorted({r["data_source"] for r in group})
        if len(values) < 2:
            continue
        same_report = len({r["publication_file"] for r in group}) == 1
        if same_report:
            labels = sorted({r["source_label"] for r in group})
            detail = "; ".join(f"{r['r1_value']} ({r['source_label']})" for r in group)
            for rec in group:
                rec["notes"].append(
                    "same 2D-VNT report gives more than one r1 for this serum strain; "
                    f"source labels: {', '.join(labels)}. Both rows kept"
                )
            contradiction_rows.append({
                "virus_name": virus,
                "serum_name": serum,
                "conflict_type": "same_report_multiple_r1",
                "r1_values": "; ".join(f"{v:g}" for v in values),
                "data_sources": " | ".join(sources),
                "n_rows_kept": len(group),
                "detail": detail,
            })
            continue
        summary = "; ".join(f"{r['data_source']}={r['r1_value']}" for r in group)
        for rec in group:
            others = [
                f"{r['data_source']} r1={r['r1_value']}"
                for r in group
                if r["data_source"] != rec["data_source"] or r["r1_value"] != rec["r1_value"]
            ]
            rec["notes"].append(
                "CONTRADICTION: same virus-serum pair has another 2D-VNT r1 in "
                + "; ".join(others)
            )
        contradiction_rows.append({
            "virus_name": virus,
            "serum_name": serum,
            "conflict_type": "cross_source_2d_vnt",
            "r1_values": "; ".join(f"{v:g}" for v in values),
            "data_sources": " | ".join(sources),
            "n_rows_kept": len(group),
            "detail": summary,
        })

    discordance_rows = []
    for key in sorted(set(vnt_by_pair) & set(other_by_pair)):
        vnt_vals = sorted({float(r["r1_value"]) for r in vnt_by_pair[key]})
        other_vals = sorted({float(r["r1_value"]) for r in other_by_pair[key]})
        if vnt_vals == other_vals:
            continue
        virus, serum = key
        discordance_rows.append({
            "virus_name": virus,
            "serum_name": serum,
            "vnt_r1_values": "; ".join(f"{v:g}" for v in vnt_vals),
            "other_assay_r1_values": "; ".join(f"{v:g}" for v in other_vals),
            "vnt_sources": " | ".join(sorted({r["data_source"] for r in vnt_by_pair[key]})),
            "other_sources": " | ".join(sorted({r["data_source"] for r in other_by_pair[key]})),
            "note": "Different assay. The non-VNT value is excluded from the master and is not a second 2D-VNT measurement.",
        })

    included = []
    excluded = []
    for rec in deduped:
        notes = "; ".join(dict.fromkeys(n for n in rec["notes"] if n))
        out = {
            "virus_name": rec["virus_name"],
            "virus_accession": rec["virus_accession"],
            "serum_name": rec["serum_name"],
            "r1_value": rec["r1_value"],
            "r1_binary": rec["r1_binary"],
            "lineage": rec["lineage"],
            "sub_lineage": rec["sub_lineage"],
            "country": rec["country"],
            "year_isolated": rec["year_isolated"],
            "host_species": rec["host_species"],
            "VNT_method": rec["VNT_method"],
            "data_source": rec["data_source"],
            "data_format": rec["data_format"],
            "notes": notes,
        }
        if rec["exclusion_reasons"]:
            excluded.append({
                **out,
                "r1_as_published": (
                    f"{rec['r1_operator']}{rec['r1_text']}" if rec["r1_text"] else ""
                ),
                "exclusion_reasons": ";".join(rec["exclusion_reasons"]),
            })
        else:
            included.append(out)

    included.sort(key=lambda r: (
        r["lineage"], r["sub_lineage"], r["virus_name"], r["serum_name"],
        r["data_source"], r["r1_value"],
    ))
    for i, rec in enumerate(included, 1):
        rec["pair_id"] = f"P{i:04d}"

    master_fields = [
        "pair_id", "virus_name", "virus_accession", "serum_name",
        "r1_value", "r1_binary", "lineage", "sub_lineage",
        "country", "year_isolated", "host_species",
        "VNT_method", "data_source", "data_format", "notes",
    ]
    excluded_fields = [
        "virus_name", "virus_accession", "serum_name",
        "r1_as_published", "r1_binary", "lineage", "sub_lineage",
        "country", "year_isolated", "host_species",
        "VNT_method", "data_source", "data_format",
        "exclusion_reasons", "notes",
    ]
    contradiction_fields = [
        "virus_name", "serum_name", "conflict_type", "r1_values",
        "data_sources", "n_rows_kept", "detail",
    ]
    discordance_fields = [
        "virus_name", "serum_name", "vnt_r1_values", "other_assay_r1_values",
        "vnt_sources", "other_sources", "note",
    ]

    write_csv(DATA / "master_pairs.csv", included, master_fields)
    write_csv(DATA / "master_excluded.csv", excluded, excluded_fields)
    write_csv(DATA / "master_contradictions.csv", contradiction_rows, contradiction_fields)
    write_csv(DATA / "master_assay_discordance.csv", discordance_rows, discordance_fields)

    # Sequential funnel, in the order of the inclusion criteria.
    n = len(deduped)
    step_vnt = [r for r in deduped if "not_2d_vnt" not in r["exclusion_reasons"]]
    step_r1 = [
        r for r in step_vnt
        if "r1_missing" not in r["exclusion_reasons"]
        and "r1_censored" not in r["exclusion_reasons"]
    ]
    step_seq = [
        r for r in step_r1
        if "no_vp1_or_p1_sequence" not in r["exclusion_reasons"]
    ]

    from collections import Counter
    reason_counts = Counter()
    for rec in deduped:
        for reason in rec["exclusion_reasons"]:
            reason_counts[reason] += 1

    iran = [r for r in included if r["lineage"] == "A/ASIA/Iran-05"]
    sub_counts = Counter(r["sub_lineage"] or "(blank)" for r in iran)
    focus = ["SIS-10", "SIS-12", "FAR-11", "BAR-08", "ARD-07", "AFG-07", "SIS-13"]
    catalogue_focus = Counter()
    for virus in viruses:
        sub = norm_sublineage(virus.get("sublineage"))
        if sub in focus:
            catalogue_focus[sub] += 1
    master_viruses = {r["virus_name"] for r in included}
    focus_in_master = Counter()
    focus_missing = defaultdict(list)
    for virus in viruses:
        sub = norm_sublineage(virus.get("sublineage"))
        if sub not in focus:
            continue
        name = clean(virus["field_virus"])
        if name in master_viruses:
            focus_in_master[sub] += 1
        else:
            focus_missing[sub].append(name)

    lines = [
        "FMDV serotype A master pair table",
        "",
        f"Input pairs after dropping exact duplicate rows: {n}",
        f"Master rows (all inclusion criteria met): {len(included)}",
        f"Excluded rows: {len(excluded)}",
        f"2D-VNT r1 disagreements (both rows kept): {len(contradiction_rows)}",
        f"    of which cross-source: {sum(1 for r in contradiction_rows if r['conflict_type']=='cross_source_2d_vnt')}",
        f"    of which same-report multiple labels: {sum(1 for r in contradiction_rows if r['conflict_type']=='same_report_multiple_r1')}",
        f"VNT versus LPBE/ELISA disagreements (LPBE excluded from master): {len(discordance_rows)}",
        "",
        "Inclusion funnel",
        f"1. All gathered pairs: {n}",
        f"2. 2D-VNT (LPBE/ELISA removed): {len(step_vnt)}",
        "3. Bovine vaccinate serum: "
        f"{len(step_vnt)} (WRL matching reports and Singanallur 2022 use bovine vaccinate serum; "
        "Azimi 2020 states bovine vaccine sera. No guinea-pig or mouse serum was present.)",
        f"4. Exact numeric r1 (missing and censored bounds removed): {len(step_r1)}",
        f"5. VP1 or P1 sequence in the local GenBank extract: {len(step_seq)}",
        "",
        "Rows can fail more than one rule. Counts below are not a funnel:",
    ]
    for reason, count in reason_counts.most_common():
        lines.append(f"- {reason}: {count}")
    lines += [
        "",
        f"Iran-05 rows in the master: {len(iran)}",
        f"Iran-05 viruses in the master: {len({r['virus_name'] for r in iran})}",
        "Iran-05 sublineage row counts in the master:",
    ]
    for sub, count in sub_counts.most_common():
        lines.append(f"- {sub}: {count}")
    lines.append("")
    lines.append("Focus sublineages (catalogue viruses vs viruses that reached the master):")
    for sub in focus:
        lines.append(
            f"- {sub}: catalogue {catalogue_focus[sub]}, in master {focus_in_master[sub]}"
        )
        missing = focus_missing.get(sub) or []
        if missing:
            lines.append("  not in master: " + ", ".join(missing))
    lines += [
        "",
        "data_format",
        "- table: exact r1 from a WRL report table, a paper table, or the PDF text layer.",
        "- digitized: unused. No WebPlotDigitizer values are in the gathered files.",
        "- personal_communication: unused.",
        "",
        "Not entered, because the gathered files do not contain an exact r1 joined to a named virus:",
        "- Upadhyaya 2014 and Bari 2014 bar charts (virus lists are in viruses.csv; r1 was not numeric in a table).",
        "- Azimi 2020 Cluster 1: six r1 values are not one-to-one with the seven isolate names (see unresolved_r1.csv). Qom B3 = 0.19 was gathered but has no VP1/P1 accession, so it is excluded.",
        "- Scanned WRL reports listed in scanned_reports_not_digitized.txt, including most Iran reports from 2009-2021.",
        "",
        "Censored r1 values such as <0.17 or >1.00 are in master_excluded.csv. They were not treated as point estimates.",
        "2D-VNT disagreements are in master_contradictions.csv. Both rows stay in the master when they also meet the sequence rule.",
        "A same-report disagreement is two serum labels of one vaccine strain (for example A22 IRQ and A22-550), not two publications.",
        "LPBE/ELISA numbers that differ from a 2D-VNT number are in master_assay_discordance.csv. They are not copied into the master as a second r1.",
        "",
        "Column notes is the audit trail required when a value was standardized or two sources disagree.",
        "r1_binary is 1 when r1 >= 0.3 and 0 when r1 < 0.3.",
    ]
    (DATA / "master_audit.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
