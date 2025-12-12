#!/usr/bin/env python3
"""
DNA test data generator for parallel string-matching experiments.

Features:
- DNA alphabet: A, C, G, T
- Generates:
  1) Match-density cases (100 densities from low to high)
  2) Pattern-length cases
  3) Front-vs-back match placement cases
- Streams to disk in chunks so it can handle very long texts (e.g., 1e12).

Output format:
- Each case writes two files into an output directory:
  - {case_name}_pattern.txt
  - {case_name}_text.txt

"""

import os
import argparse
import math
import random
from typing import Iterable

DNA_ALPHABET = "ACGT"


# ---------- Low-level utilities ----------

def random_dna_pattern(length: int, rng: random.Random) -> str:
    """Generate a random DNA pattern of given length."""
    return "".join(rng.choice(DNA_ALPHABET) for _ in range(length))


def random_base_excluding(exclude: str, rng: random.Random) -> str:
    """Return a random base not equal to `exclude`."""
    choices = [b for b in DNA_ALPHABET if b != exclude]
    return rng.choice(choices)


def write_text_stream(
    filepath: str,
    total_len: int,
    pattern: str,
    match_positions: Iterable[int],
    rng: random.Random,
    chunk_size: int = 10**7,
):
    """
    Stream-generate a DNA text of length total_len, writing directly to `filepath`.

    - `pattern` is the exact match string to embed.
    - `match_positions` is an iterable of starting indices where the pattern must appear.
    - Other positions are filled with random DNA (may incidentally contain extra matches).
    - Uses chunked writing to support very large texts without high memory usage.
    """
    pattern_len = len(pattern)
    match_positions = sorted(match_positions)
    match_idx = 0
    num_matches = len(match_positions)

    # Precompute a set for O(1) checks (positions where pattern starts)
    match_pos_set = set(match_positions)

    with open(filepath, "w", encoding="utf-8") as f:
        pos = 0
        while pos < total_len:
            # We build a chunk in memory but keep it reasonably small.
            chunk_end = min(total_len, pos + chunk_size)
            buf = []

            while pos < chunk_end:
                if pos in match_pos_set:
                    # Emit the pattern here
                    buf.append(pattern)
                    pos += pattern_len
                else:
                    # Emit a single random base here
                    buf.append(rng.choice(DNA_ALPHABET))
                    pos += 1

            f.write("".join(buf))


def safe_num_matches(text_len: int, pattern_len: int, desired_density: float) -> int:
    """
    Decide how many matches we *try* to embed for a given density.

    Here density is defined per possible starting position: number_of_matches /
    (text_len - pattern_len + 1).
    """
    max_starts = max(1, text_len - pattern_len + 1)
    n = int(desired_density * max_starts)
    return max(0, min(n, max_starts))


# ---------- Case generators ----------

def generate_match_density_cases(
    out_dir: str,
    text_len: int,
    pattern_len: int,
    num_cases: int = 100,
    min_density: float = 1e-6,
    max_density: float = 1e-1,
    seed: int = 42,
):
    """
    Generate `num_cases` datasets with different match densities.
    Density ranges from min_density to max_density (inclusive, approx. linear in log-space).

    All matches are roughly spread from the front (non-random positions);
    position effect is not the focus here.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)

    # Log-spaced densities so we cover low to high more evenly
    densities = []
    for i in range(num_cases):
        t = i / (num_cases - 1) if num_cases > 1 else 0.0
        # log-space interpolation
        d = min_density * (max_density / min_density) ** t
        densities.append(d)

    pattern = random_dna_pattern(pattern_len, rng)

    # Write pattern once, reused by all density cases
    pattern_path = os.path.join(out_dir, "density_pattern_len{}_pattern.txt".format(pattern_len))
    with open(pattern_path, "w", encoding="utf-8") as f:
        f.write(pattern)

    for idx, density in enumerate(densities):
        num_matches = safe_num_matches(text_len, pattern_len, density)

        # Place matches in consecutive starting positions from the beginning
        starts = list(range(num_matches))  # overlapping matches: 0,1,2,...

        case_name = f"density_case_{idx:03d}_d{density:.2e}"
        text_path = os.path.join(out_dir, case_name + "_text.txt")

        print(f"[density] Generating {case_name}: text_len={text_len}, "
              f"pattern_len={pattern_len}, density≈{density:.2e}, matches={num_matches}")

        write_text_stream(
            filepath=text_path,
            total_len=text_len,
            pattern=pattern,
            match_positions=starts,
            rng=rng,
        )


def generate_pattern_length_cases(
    out_dir: str,
    text_len: int,
    pattern_lengths,
    target_density: float = 1e-4,
    seed: int = 123,
):
    """
    For each pattern length in `pattern_lengths`, generate one dataset.
    All use the same text_len and target match density.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)

    for plen in pattern_lengths:
        pattern = random_dna_pattern(plen, rng)
        num_matches = safe_num_matches(text_len, plen, target_density)
        starts = list(range(num_matches))  # overlapping from front

        case_name = f"plen_{plen}"
        pattern_path = os.path.join(out_dir, case_name + "_pattern.txt")
        text_path = os.path.join(out_dir, case_name + "_text.txt")

        with open(pattern_path, "w", encoding="utf-8") as f:
            f.write(pattern)

        print(f"[plen] Generating {case_name}: text_len={text_len}, "
              f"pattern_len={plen}, density≈{target_density:.2e}, matches={num_matches}")

        write_text_stream(
            filepath=text_path,
            total_len=text_len,
            pattern=pattern,
            match_positions=starts,
            rng=rng,
        )


def generate_front_back_cases(
    out_dir: str,
    text_len: int,
    pattern_len: int,
    density: float = 1e-4,
    seed: int = 999,
):
    """
    Generate two datasets with the same pattern/density:
    - One where all matches are packed at the front.
    - One where all matches are packed at the back.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)

    pattern = random_dna_pattern(pattern_len, rng)
    num_matches = safe_num_matches(text_len, pattern_len, density)

    pattern_path = os.path.join(out_dir, "front_back_pattern_len{}_pattern.txt".format(pattern_len))
    with open(pattern_path, "w", encoding="utf-8") as f:
        f.write(pattern)

    # Front: contiguous matches from position 0
    front_starts = list(range(num_matches))

    # Back: contiguous matches ending at text_len - 1
    # using overlapping starts: last start at text_len - pattern_len
    max_start = text_len - pattern_len
    back_starts = list(range(max_start - num_matches + 1, max_start + 1))

    # Front case
    print(f"[front/back] Generating front case: text_len={text_len}, "
          f"pattern_len={pattern_len}, density≈{density:.2e}, matches={num_matches}")
    front_text_path = os.path.join(out_dir, "front_matches_text.txt")
    write_text_stream(
        filepath=front_text_path,
        total_len=text_len,
        pattern=pattern,
        match_positions=front_starts,
        rng=rng,
    )

    # Back case
    print(f"[front/back] Generating back case: text_len={text_len}, "
          f"pattern_len={pattern_len}, density≈{density:.2e}, matches={num_matches}")
    back_text_path = os.path.join(out_dir, "back_matches_text.txt")
    write_text_stream(
        filepath=back_text_path,
        total_len=text_len,
        pattern=pattern,
        match_positions=back_starts,
        rng=rng,
    )


# ---------- CLI ----------

def parse_args():
    parser = argparse.ArgumentParser(
        description="DNA test data generator for string-matching experiments."
    )

    subparsers = parser.add_subparsers(dest="mode", required=True)

    # 1) match-density mode
    p_density = subparsers.add_parser("density", help="Generate 100 match-density cases")
    p_density.add_argument("--out_dir", type=str, required=True)
    p_density.add_argument("--text_len", type=int, default=10**8,
                           help="Length of the text (e.g. 10**12 on your cluster)")
    p_density.add_argument("--pattern_len", type=int, default=64)
    p_density.add_argument("--num_cases", type=int, default=100)
    p_density.add_argument("--min_density", type=float, default=1e-6)
    p_density.add_argument("--max_density", type=float, default=1e-1)
    p_density.add_argument("--seed", type=int, default=42)

    # 2) pattern-length mode
    p_plen = subparsers.add_parser("plen", help="Generate pattern-length vs throughput cases")
    p_plen.add_argument("--out_dir", type=str, required=True)
    p_plen.add_argument("--text_len", type=int, default=10**8)
    p_plen.add_argument(
        "--pattern_lengths",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64, 128, 256, 512, 1024],
        help="List of pattern lengths."
    )
    p_plen.add_argument("--target_density", type=float, default=1e-4)
    p_plen.add_argument("--seed", type=int, default=123)

    # 3) front-vs-back mode
    p_fb = subparsers.add_parser("frontback", help="Generate front-vs-back match placement cases")
    p_fb.add_argument("--out_dir", type=str, required=True)
    p_fb.add_argument("--text_len", type=int, default=10**8)
    p_fb.add_argument("--pattern_len", type=int, default=64)
    p_fb.add_argument("--density", type=float, default=1e-4)
    p_fb.add_argument("--seed", type=int, default=999)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "density":
        generate_match_density_cases(
            out_dir=args.out_dir,
            text_len=args.text_len,
            pattern_len=args.pattern_len,
            num_cases=args.num_cases,
            min_density=args.min_density,
            max_density=args.max_density,
            seed=args.seed,
        )
    elif args.mode == "plen":
        generate_pattern_length_cases(
            out_dir=args.out_dir,
            text_len=args.text_len,
            pattern_lengths=args.pattern_lengths,
            target_density=args.target_density,
            seed=args.seed,
        )
    elif args.mode == "frontback":
        generate_front_back_cases(
            out_dir=args.out_dir,
            text_len=args.text_len,
            pattern_len=args.pattern_len,
            density=args.density,
            seed=args.seed,
        )
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
