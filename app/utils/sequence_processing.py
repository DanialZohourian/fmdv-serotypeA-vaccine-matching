# -*- coding: utf-8 -*-
"""Validate a VP1 sequence, place it on the 213-column reference, and build model features."""
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from Bio.Align import PairwiseAligner

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
SKIP_CHARS = {"-", "X"}
MOTIF = re.compile(r"N[^P][ST]")
ALIGNER = PairwiseAligner()
ALIGNER.mode = "global"
ALIGNER.match_score = 2
ALIGNER.mismatch_score = -1
ALIGNER.open_gap_score = -5
ALIGNER.extend_gap_score = -0.5


def validate_sequence(seq: str) -> tuple[bool, str]:
    """Return (True, cleaned sequence) or (False, error message)."""
    seq = seq.strip().upper()
    if seq.startswith(">"):
        seq = "".join(line.strip() for line in seq.splitlines() if line and not line.startswith(">"))
    seq = seq.replace(" ", "").replace("\n", "").replace("\r", "")
    if len(seq) == 0:
        return False, "The sequence is empty."
    invalid = set(seq) - VALID_AA - SKIP_CHARS
    if invalid:
        return False, f"Characters not allowed in a protein sequence: {''.join(sorted(invalid))}"
    ungapped = seq.replace("-", "").replace("X", "")
    if not (208 <= len(ungapped) <= 220):
        return False, f"{len(ungapped)} amino acids. A VP1 protein should be 208–220 amino acids."
    return True, seq.replace("-", "")


def read_fasta(path) -> list[tuple[str, str]]:
    records = []
    header = None
    chunks = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(chunks)))
            header = line[1:]
            chunks = []
        elif line.strip():
            chunks.append(line.strip().upper())
    if header is not None:
        records.append((header, "".join(chunks)))
    return records


def alignment_cache(records: list[tuple[str, str]]) -> dict[str, str]:
    """Map a gap-free sequence onto its original 213-column MAFFT row."""
    cache = {}
    conflicts = set()
    for _header, aligned in records:
        key = aligned.replace("-", "")
        previous = cache.get(key)
        if previous is not None and previous != aligned:
            conflicts.add(key)
        else:
            cache[key] = aligned
    for key in conflicts:
        del cache[key]
    return cache


def find_mafft() -> tuple[str, Path] | None:
    """Locate MAFFT. Returns ("native", path) on Linux/macOS or ("winbash", path) on Windows.

    Resolution order:
      1. MAFFT_PATH  - a native mafft executable (Linux, macOS, WSL, Streamlit Cloud)
      2. mafft on PATH
      3. MAFFT_BASH  - bash.exe shipped with mafft-win, for the Windows all-in-one build
    """
    explicit = os.environ.get("MAFFT_PATH")
    if explicit and Path(explicit).exists():
        return ("native", Path(explicit))
    found = shutil.which("mafft")
    if found:
        return ("native", Path(found))
    win_bash = os.environ.get("MAFFT_BASH")
    if win_bash and Path(win_bash).exists():
        return ("winbash", Path(win_bash))
    return None


def mafft_project(queries: list[str], profile: list[str]) -> list[str]:
    """Add sequences to the fixed 213-column alignment with MAFFT --addfragments."""
    located = find_mafft()
    if located is None:
        raise FileNotFoundError(
            "MAFFT was not found. Install it and make sure it is on the PATH, or set "
            "MAFFT_PATH (Linux/macOS) or MAFFT_BASH (Windows mafft-win build)."
        )
    flavour, mafft_path = located
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        reference = folder / "reference.fasta"
        query_file = folder / "query.fasta"
        output = folder / "added.fasta"
        reference.write_text(
            "".join(f">r{i}\n{seq}\n" for i, seq in enumerate(profile)),
            encoding="utf-8",
        )
        query_file.write_text(
            "".join(f">q{i}\n{seq}\n" for i, seq in enumerate(queries)),
            encoding="utf-8",
        )
        env = os.environ.copy()
        env["MAFFT_TMPDIR"] = str(folder)
        arguments = [
            "--addfragments", str(query_file), "--keeplength", "--quiet", str(reference),
        ]
        if flavour == "native":
            command = [str(mafft_path)] + arguments
        else:
            # Windows mafft-win all-in-one build: mafft is a shell script run under its own bash.
            env["ROOTDIR"] = str(mafft_path.parents[2]) + "\\"
            env["MAFFT_BINARIES"] = "/usr/lib/mafft"
            env["PATH"] = "/usr/bin/;" + env.get("PATH", "")
            command = [str(mafft_path), "/usr/bin/mafft"] + arguments
        with output.open("w", encoding="utf-8") as handle:
            subprocess.run(
                command,
                check=True, stdout=handle, stderr=subprocess.DEVNULL, env=env, timeout=180,
            )
        found = {}
        header = None
        chunks = []
        for line in output.read_text(encoding="utf-8").splitlines():
            if line.startswith(">"):
                if header and header.startswith("q"):
                    found[header] = "".join(chunks)
                header = line[1:].split()[0]
                chunks = []
            elif line.strip():
                chunks.append(line.strip().upper())
        if header and header.startswith("q"):
            found[header] = "".join(chunks)
    placed = []
    for i in range(len(queries)):
        seq = found.get(f"q{i}")
        if not seq or len(seq) != 213:
            raise ValueError("MAFFT did not return a 213-column alignment.")
        placed.append(seq)
    return placed


def align_pairwise(new_seq: str, ref_aligned: str) -> str:
    """Project a new VP1 onto the columns of one reference sequence."""
    if len(ref_aligned) != 213:
        raise ValueError(f"The reference alignment has {len(ref_aligned)} columns; 213 are required.")
    ref_nogap = ref_aligned.replace("-", "").replace("X", "")
    query = new_seq.replace("-", "").replace(" ", "")
    alignment = next(iter(ALIGNER.align(ref_nogap, query)), None)
    if alignment is None:
        raise ValueError("The sequence could not be aligned. It may not be VP1.")
    ref_aln = str(alignment[0])
    new_aln = str(alignment[1])
    if ref_aln.replace("-", "") != ref_nogap:
        raise ValueError("Alignment changed the reference sequence.")

    ref_to_new = {}
    ref_index = 0
    for ref_aa, new_aa in zip(ref_aln, new_aln):
        if ref_aa != "-":
            ref_to_new[ref_index] = new_aa
            ref_index += 1
        elif new_aa == "-":
            raise ValueError("The alignment contained an empty column.")

    result = []
    ref_index = 0
    for ref_aa in ref_aligned:
        if ref_aa not in SKIP_CHARS:
            result.append(ref_to_new.get(ref_index, "-"))
            ref_index += 1
        else:
            result.append("-")
    aligned = "".join(result)
    comparable = sum(
        1 for a, b in zip(aligned, ref_aligned) if a not in SKIP_CHARS and b not in SKIP_CHARS
    )
    if comparable < 150:
        raise ValueError("The sequence could not be aligned. It may not be VP1.")
    return aligned


def place_sequences(sequences: list[str], ref_aligned: str, cache: dict[str, str], profile: list[str]) -> tuple[list[str], str]:
    """Place sequences on the reference columns. Known sequences use the MAFFT row."""
    placed = [None] * len(sequences)
    pending = []
    for index, seq in enumerate(sequences):
        key = seq.replace("-", "").replace(" ", "").upper()
        known = cache.get(key)
        if known is not None:
            placed[index] = known
        else:
            pending.append((index, key))
    note = ""
    if pending:
        try:
            added = mafft_project([seq for _index, seq in pending], profile)
            for (index, _seq), aligned in zip(pending, added):
                placed[index] = aligned
            note = "New sequence placed on the 213-column alignment with MAFFT."
        except (OSError, subprocess.SubprocessError, ValueError, FileNotFoundError):
            for index, seq in pending:
                placed[index] = align_pairwise(seq, ref_aligned)
            note = "MAFFT was not available, so a pairwise alignment was used. G-H loop columns can shift relative to the model."
    return placed, note


def align_to_reference(new_seq: str, ref_aligned: str, cache: dict | None = None, profile: list[str] | None = None) -> str:
    placed, _note = place_sequences([new_seq], ref_aligned, cache or {}, profile or [ref_aligned])
    return placed[0]


def poisson_corrected_distance(seq1: str, seq2: str) -> float:
    """Same Poisson-corrected p-distance used to train the model."""
    comparable = [(a, b) for a, b in zip(seq1, seq2) if a not in SKIP_CHARS and b not in SKIP_CHARS]
    if not comparable:
        raise ValueError("The two sequences have no comparable positions.")
    p = sum(a != b for a, b in comparable) / len(comparable)
    if p == 0:
        return 0.0
    if p >= 1:
        raise ValueError("The two sequences differ at every comparable position.")
    return float(-np.log(1.0 - p))


def find_nglyc_sites(seq_aligned: str) -> set[int]:
    seq_nogap = seq_aligned.replace("-", "").replace("X", "")
    return {match.start() + 1 for match in MOTIF.finditer(seq_nogap)}


def extract_features(virus_seq_aligned: str, serum_seq_aligned: str, selected_features: list[str]) -> dict:
    """Calculate the 23 model features from two sequences in alignment columns."""
    features = {"pairwise_aa_distance": poisson_corrected_distance(virus_seq_aligned, serum_seq_aligned)}
    for feature in selected_features:
        if not feature.startswith("site_"):
            continue
        position = int(feature.split("_")[1]) - 1
        virus_aa = virus_seq_aligned[position]
        serum_aa = serum_seq_aligned[position]
        if virus_aa in SKIP_CHARS or serum_aa in SKIP_CHARS:
            features[feature] = 0.0
        else:
            features[feature] = 0.0 if virus_aa == serum_aa else 1.0
    virus_sites = find_nglyc_sites(virus_seq_aligned)
    serum_sites = find_nglyc_sites(serum_seq_aligned)
    features["nglyc_count_diff"] = float(abs(len(virus_sites) - len(serum_sites)))
    features["nglyc_shared"] = float(len(virus_sites & serum_sites))
    features["nglyc_virus_only"] = float(len(virus_sites - serum_sites))
    features["nglyc_serum_only"] = float(len(serum_sites - virus_sites))
    missing = [feature for feature in selected_features if feature not in features]
    if missing:
        raise KeyError(f"Missing features: {missing}")
    return features


def features_to_array(features: dict, selected_features: list[str]) -> np.ndarray:
    return np.array([[features[feature] for feature in selected_features]], dtype=np.float64)


def match_shap(explainer, matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """SHAP contributions for the match class, plus the base value on the probability scale."""
    values = explainer.shap_values(matrix, check_additivity=False)
    expected = explainer.expected_value
    if hasattr(values, "values"):
        expected = values.base_values
        values = values.values
    array = np.asarray(values)
    base = np.atleast_1d(np.asarray(expected, dtype=float))
    if array.ndim == 2:
        return array[0], float(base.reshape(-1)[-1])
    if array.ndim == 3 and array.shape[-1] == 2:
        return array[0, :, 1], float(base.reshape(-1)[-1])
    if array.ndim == 3 and array.shape[1] == 2:
        return array[0, 1, :], float(base.reshape(-1)[-1])
    raise RuntimeError(f"Unexpected SHAP output shape: {array.shape}")
