# -*- coding: utf-8 -*-
"""Align the 81 VP1 proteins with MAFFT L-INS-i and trim gappy columns."""
import csv
import subprocess
from collections import Counter
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "step2_sequences" / "05_final" / "vp1_all_protein.fasta"
OUT = ROOT / "step3_alignment"
MAFFT = Path(os.environ.get("MAFFT_PATH", shutil.which("mafft") or "mafft"))
VALID = set("ACDEFGHIKLMNPQRSTVWY")
SPACE = "___"


def read_fasta(path):
    records = []
    header = None
    chunks = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(chunks)))
            header = line[1:].strip()
            chunks = []
        elif line.strip():
            chunks.append(line.strip())
    if header is not None:
        records.append((header, "".join(chunks)))
    return records


def write_fasta(path, records, width=60):
    lines = []
    for header, seq in records:
        lines.append(">" + header)
        for i in range(0, len(seq), width):
            lines.append(seq[i:i + width])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate(records):
    issues = []
    for header, seq in records:
        if not (208 <= len(seq) <= 220):
            issues.append(f"unusual length: {header.split('|')[0]} ({len(seq)} aa)")
        invalid = set(seq.upper()) - VALID - {"-", "X", "B", "Z"}
        if invalid:
            issues.append(f"invalid characters: {header.split('|')[0]} ({invalid})")
        if "|" not in header:
            issues.append(f"header has no pipe: {header}")
    return issues


def restore(header, mapping):
    if header not in mapping:
        raise SystemExit(f"MAFFT returned an unknown header: {header}")
    return mapping[header]


def gap_stats(records):
    n_seqs = len(records)
    aln_length = len(records[0][1])
    if any(len(seq) != aln_length for _, seq in records):
        raise SystemExit("alignment rows have different lengths")
    gap_frac = []
    for i in range(aln_length):
        gaps = sum(seq[i] == "-" for _, seq in records)
        gap_frac.append(gaps / n_seqs)
    return gap_frac


def rgd_columns(records):
    columns = []
    missing = []
    for header, seq in records:
        name = header.split("|")[0]
        ungapped = seq.replace("-", "")
        at = ungapped.find("RGD")
        if at < 0:
            missing.append(name)
            continue
        seen = 0
        col = None
        for i, char in enumerate(seq):
            if char == "-":
                continue
            if seen == at:
                col = i
                break
            seen += 1
        columns.append((name, col, seq[col:col + 3]))
    return columns, missing


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = read_fasta(SRC)
    print(f"sequences: {len(records)}")
    issues = validate(records)
    if issues:
        for issue in issues:
            print("ISSUE", issue)
        raise SystemExit(1)
    print("input sequences passed length, character, and header checks")
    lengths = Counter(len(seq) for _, seq in records)
    for length, count in sorted(lengths.items()):
        print(f"  {length} aa: {count}")
    types = Counter(header.split("|")[-1] for header, _ in records)
    print("types", dict(types))

    mapping = {}
    sanitized = []
    for header, seq in records:
        safe = header.replace(" ", SPACE)
        if safe in mapping:
            raise SystemExit(f"header collision after removing spaces: {header}")
        mapping[safe] = header
        sanitized.append((safe, seq))
    safe_input = OUT / "_input_headers_sanitized.fasta"
    write_fasta(safe_input, sanitized)

    raw_path = OUT / "aligned_vp1_protein_raw.fasta"
    command = [
        "cmd.exe", "/C", str(MAFFT),
        "--localpair", "--maxiterate", "1000", "--reorder", "--amino",
        "--thread", "-1",
        "--out", str(raw_path),
        str(safe_input),
    ]
    print("running MAFFT L-INS-i")
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    log_path = OUT / "mafft_stderr.txt"
    log_path.write_text((completed.stderr or "") + "\n" + (completed.stdout or ""), encoding="utf-8")
    if completed.returncode != 0 or not raw_path.exists():
        print(completed.stdout[-1000:])
        print(completed.stderr[-1000:])
        raise SystemExit(f"MAFFT failed with code {completed.returncode}")
    print("MAFFT finished")
    print((completed.stderr or "")[:500])

    aligned = [(restore(header, mapping), seq.upper()) for header, seq in read_fasta(raw_path)]
    write_fasta(raw_path, aligned)
    gap_frac = gap_stats(aligned)
    n_seqs = len(aligned)
    aln_length = len(aligned[0][1])
    print(f"raw sequences: {n_seqs}")
    print(f"raw alignment length: {aln_length}")
    print(f"columns with no gap: {sum(v == 0 for v in gap_frac)}")
    print(f"columns with <10% gap: {sum(v < 0.1 for v in gap_frac)}")
    print(f"columns with 10-50% gap: {sum(0.1 <= v < 0.5 for v in gap_frac)}")
    print(f"columns with >50% gap: {sum(v >= 0.5 for v in gap_frac)}")
    print(f"columns with 100% gap: {sum(v == 1 for v in gap_frac)}")
    high_gap = []
    for header, seq in aligned:
        pct = seq.count("-") / len(seq) * 100
        if pct > 10:
            high_gap.append(f"{header.split('|')[0]}: {pct:.1f}%")
    print("sequences with >10% gaps:", high_gap or "none")

    columns, missing = rgd_columns(aligned)
    used = Counter(col for _, col, _ in columns)
    print("RGD alignment columns (0-based):", dict(used))
    print("sequences without RGD:", missing or "none")
    if len(used) == 1:
        col = next(iter(used))
        window = aligned[0][1][max(0, col - 7):col + 18]
        print(f"G-H window around RGD column {col + 1}: consensus slice starts at {max(1, col - 6)}")
        varied = 0
        for offset in range(max(0, col - 7), min(aln_length, col + 18)):
            letters = {seq[offset] for _, seq in aligned}
            if len(letters - {"-"}) > 1:
                varied += 1
        print(f"variable columns in the 25-column G-H window: {varied}")

    keep = [i for i, frac in enumerate(gap_frac) if frac <= 0.5]
    removed = [i for i, frac in enumerate(gap_frac) if frac > 0.5]
    print(f"kept columns: {len(keep)} of {aln_length}")
    print(f"removed columns (>50% gap): {removed}")
    for i in removed:
        residues = Counter(seq[i] for _, seq in aligned if seq[i] != "-")
        names = [header.split("|")[0] for header, seq in aligned if seq[i] != "-"]
        print(f"  column {i + 1}: {dict(residues)} in {names}")

    trimmed = []
    for header, seq in aligned:
        trimmed.append((header, "".join(seq[i] for i in keep)))
    trimmed_path = OUT / "aligned_vp1_protein_trimmed.fasta"
    write_fasta(trimmed_path, trimmed)
    print(f"trimmed length: {len(keep)}")

    report_path = OUT / "alignment_report.csv"
    rows = []
    for header, seq in trimmed:
        parts = header.split("|")
        gap_count = seq.count("-")
        gap_pct = gap_count / len(seq) * 100
        start = next(i for i, char in enumerate(seq) if char != "-")
        end = len(seq) - next(i for i, char in enumerate(reversed(seq)) if char != "-")
        coverage = (end - start) / len(seq) * 100
        rows.append({
            "seq_id": parts[0],
            "type": parts[-1] if len(parts) > 4 else "unknown",
            "aligned_length": len(seq),
            "original_length": len(seq) - gap_count,
            "gap_count": gap_count,
            "gap_percent": round(gap_pct, 2),
            "start_pos": start + 1,
            "end_pos": end,
            "coverage_pct": round(coverage, 2),
            "status": "PASS" if gap_pct < 10 and coverage > 90 else "FLAG",
        })
    with report_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    status = Counter(row["status"] for row in rows)
    print(f"report PASS {status['PASS']} FLAG {status['FLAG']}")
    print(f"mean gap% {sum(row['gap_percent'] for row in rows) / len(rows):.2f}")
    print(f"max gap% {max(row['gap_percent'] for row in rows):.2f}")
    for row in rows:
        if row["status"] == "FLAG":
            print("FLAG", row["seq_id"], row["gap_percent"], row["coverage_pct"])

    final_lengths = {len(seq) for _, seq in trimmed}
    final_types = Counter(header.split("|")[-1] for header, _ in trimmed)
    all_gap = [header for header, seq in trimmed if seq.replace("-", "") == ""]
    bad_headers = [header for header, _ in trimmed if "|" not in header]
    print(f"final count {len(trimmed)}")
    print(f"final lengths {final_lengths}")
    print(f"all-gap sequences {len(all_gap)}")
    print(f"headers without pipe {len(bad_headers)}")
    print(f"virus {final_types['virus']} serum {final_types['serum']}")
    safe_input.unlink()


if __name__ == "__main__":
    main()
