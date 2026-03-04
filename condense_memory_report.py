#!/usr/bin/env python3
"""Condense footprint memory reports by extracting and summarizing phys_footprint samples.

Usage:
    uv run python condense_memory_report.py <input_file> [output_file]

If output_file is not specified, prints to stdout.
"""

import argparse
import re
import sys
from pathlib import Path


def extract_phys_footprint_samples(lines: list[str]) -> list[tuple[int, int, int | None]]:
    """Extract phys_footprint values from the summary section.

    Returns list of (phys_footprint, phys_footprint_peak, neural_peak) in MB.
    neural_peak may be None if not available.
    """
    samples = []
    pending_neural_peak: int | None = None

    for line in lines:
        # Check for neural_peak in Auxiliary data section
        neural_match = re.search(r'neural_peak:\s+(\d+)\s+MB', line)
        if neural_match:
            pending_neural_peak = int(neural_match.group(1))

        # Match lines like: "  > phys_footprint: 266 MB"
        match = re.search(r'phys_footprint:\s+(\d+)\s+MB', line)
        peak_match = re.search(r'phys_footprint_peak:\s+(\d+)\s+MB', line)
        if match:
            phys = int(match.group(1))
            # Try to find corresponding peak on same line or nearby
            peak = phys  # default
            if peak_match:
                peak = int(peak_match.group(1))
            # Attach neural_peak if we found one before this sample
            samples.append((phys, peak, pending_neural_peak))
            # Reset pending neural_peak after attaching
            pending_neural_peak = None
    return samples


def condense_samples(samples: list[tuple[int, int, int | None]], max_samples: int = 20) -> list[tuple[int, int, int | None]]:
    """Reduce sample count by keeping only significant changes.

    Uses a simple downsampling that preserves the first, last, and
    samples where significant changes occur.
    """
    if len(samples) <= max_samples:
        return samples

    # Keep first sample
    result = [samples[0]]

    # Calculate how many samples to skip
    step = len(samples) // max_samples

    # Keep samples where value changes significantly
    prev_phys = samples[0][0]
    for i in range(1, len(samples) - 1):
        phys, peak, neural = samples[i]
        # Keep if significant change (> 10% or > 50 MB)
        change_pct = abs(phys - prev_phys) / max(prev_phys, 1) * 100
        change_abs = abs(phys - prev_phys)
        if change_pct > 10 or change_abs > 50 or i % step == 0:
            result.append((phys, peak, neural))
            prev_phys = phys

    # Keep last sample
    result.append(samples[-1])

    return result


def format_condensed_report(input_path: Path, samples: list[tuple[int, int, int | None]]) -> str:
    """Create a condensed report with just the key samples."""
    # Check if any sample has neural_peak
    has_neural = any(s[2] is not None for s in samples)

    lines = [
        f"# Condensed Memory Report: {input_path.name}",
        "",
        f"Total samples: {len(samples)}",
        f"Min phys_footprint: {min(s[0] for s in samples)} MB",
        f"Max phys_footprint: {max(s[0] for s in samples)} MB",
        f"Avg phys_footprint: {sum(s[0] for s in samples) // len(samples)} MB",
    ]

    # Add neural_peak stats if available
    neural_peaks = [s[2] for s in samples if s[2] is not None]
    if neural_peaks:
        lines.append(f"Max neural_peak: {max(neural_peaks)} MB")

    lines.extend([
        "",
        "## Key Samples (phys_footprint / phys_footprint_peak / neural_peak MB)",
        "",
    ])

    for i, (phys, peak, neural) in enumerate(samples):
        if has_neural:
            neural_str = f"{neural:4d}" if neural is not None else "   -"
            lines.append(f"  Sample {i+1:3d}: {phys:4d} MB / {peak:4d} MB / {neural_str} MB")
        else:
            lines.append(f"  Sample {i+1:3d}: {phys:4d} MB / {peak:4d} MB")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Condense footprint memory reports by extracting phys_footprint samples"
    )
    parser.add_argument("input", type=Path, help="Input memory report file")
    parser.add_argument("output", type=Path, nargs="?", help="Output file (default: stdout)")
    parser.add_argument(
        "-n", "--samples", type=int, default=20, help="Maximum number of samples to keep (default: 20)"
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: File not found: {args.input}", file=sys.stderr)
        return 1

    with open(args.input) as f:
        lines = f.readlines()

    raw_samples = extract_phys_footprint_samples(lines)
    condensed = condense_samples(raw_samples, max_samples=args.samples)
    result = format_condensed_report(args.input, condensed)

    if args.output:
        with open(args.output, "w") as f:
            f.write(result)
        print(f"Condensed report written to: {args.output}")
    else:
        print(result)

    return 0


if __name__ == "__main__":
    sys.exit(main())
