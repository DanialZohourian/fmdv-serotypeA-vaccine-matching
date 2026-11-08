# -*- coding: utf-8 -*-
"""Write the stage-3 FASTA with complete headers.

Bio.SeqIO splits a FASTA title at the first space, so names such as
'A24 Cruzeiro' and the country field 'Saudi Arabia' would be cut.
Headers are read and written as whole lines.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "aligned_vp1_protein_trimmed.fasta"
ORIGINAL = HERE.parent / "step2_sequences" / "05_final" / "vp1_all_protein.fasta"
DEST = HERE / "aligned_vp1_protein_final.fasta"

HEADER_FIXES = {
    "A": "A Sau 41/91 SI94",
    "A15": "A15 Tai 1/60",
    "A24": "A24 Cruzeiro",
}


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


def main():
    original = {header.split("|")[0]: header for header, _ in read_fasta(ORIGINAL)}
    aligned = read_fasta(SRC)
    fixed = []
    fixed_count = 0
    for header, seq in aligned:
        name = header.split("|")[0]
        # SeqIO would have left only the first word of a spaced name.
        first_word = name.split()[0]
        if first_word in HEADER_FIXES and name == first_word:
            original_name = HEADER_FIXES[first_word]
            full = original.get(original_name)
            if full is None:
                raise SystemExit(f"missing from original FASTA: {original_name}")
            print(f"fixed: '{header}' -> '{full}'")
            header = full
            fixed_count += 1
        elif name in HEADER_FIXES.values():
            print(f"already complete: {header}")
            fixed_count += 1
        fixed.append((header, seq))

    write_fasta(DEST, fixed)
    print(f"\n{fixed_count} headers confirmed")
    print(f"saved: {DEST}")

    records = read_fasta(DEST)
    bad = [header for header, _ in records if "|" not in header]
    types = [header.split("|")[-1] for header, _ in records]
    names = [header.split("|")[0] for header, _ in records]
    x_seqs = [name for (header, seq), name in zip(records, names) if "X" in seq.upper()]
    lengths = {len(seq) for _, seq in records}
    print(f"count: {len(records)} {'OK' if len(records) == 81 else 'FAIL'}")
    print(f"headers without pipe: {len(bad)} {'OK' if not bad else 'FAIL'}")
    print(f"virus: {types.count('virus')} {'OK' if types.count('virus') == 65 else 'FAIL'}")
    print(f"serum: {types.count('serum')} {'OK' if types.count('serum') == 16 else 'FAIL'}")
    print(f"one length: {lengths}")
    print(f"sequences with X: {len(x_seqs)}")
    for name in x_seqs:
        print(f"  {name}")
    for required in HEADER_FIXES.values():
        print(f"present {required}: {required in names}")


if __name__ == "__main__":
    main()
