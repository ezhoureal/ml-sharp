#!/usr/bin/env python3
"""Plot memory usage over time from footprint reports.

Usage:
    uv run python plot_memory.py <report1.txt> [<report2.txt> ...] [-o output.png]

Example:
    uv run python plot_memory.py 4bit_memory.txt original_memory.txt -o memory_comparison.png
"""

import argparse
import re
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
except ImportError:
    print("Error: matplotlib required. Install with: uv add matplotlib", file=sys.stderr)
    sys.exit(1)


def extract_memory_timeline(lines: list[str]) -> tuple[list[int], list[int], list[int], list[int | None]]:
    """Extract memory samples with their sample index as time proxy.

    Returns (sample_indices, phys_footprint, phys_footprint_peak, neural_peak) in MB.
    neural_peak may be None if not available for that sample.
    """
    samples_phys = []
    samples_peak = []
    samples_neural = []
    pending_neural = None

    for line in lines:
        # Check for neural_peak before phys_footprint
        neural_match = re.search(r'neural_peak:\s+(\d+)\s+MB', line)
        if neural_match:
            pending_neural = int(neural_match.group(1))

        phys_match = re.search(r'phys_footprint:\s+(\d+)\s+MB', line)
        peak_match = re.search(r'phys_footprint_peak:\s+(\d+)\s+MB', line)
        if phys_match:
            samples_phys.append(int(phys_match.group(1)))
            if peak_match:
                samples_peak.append(int(peak_match.group(1)))
            else:
                samples_peak.append(int(phys_match.group(1)))
            # Attach pending neural_peak if found
            samples_neural.append(pending_neural)
            pending_neural = None

    indices = list(range(len(samples_phys)))
    return indices, samples_phys, samples_peak, samples_neural


def plot_memory_reports(report_paths: list[Path], output_path: Path, sample_interval: float = 0.5) -> None:
    """Plot memory usage from multiple reports."""
    fig, ax = plt.subplots(figsize=(14, 7))

    colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']  # Green, Red, Blue, Orange
    labels = []

    for i, report_path in enumerate(report_paths):
        with open(report_path) as f:
            lines = f.readlines()

        indices, phys, peak, neural = extract_memory_timeline(lines)
        if not indices:
            print(f"Warning: No memory samples found in {report_path}", file=sys.stderr)
            continue

        # Convert sample indices to time (seconds)
        time_points = [idx * sample_interval for idx in indices]

        label = report_path.stem.replace('_memory', '')
        color = colors[i % len(colors)]

        # Plot phys_footprint
        ax.plot(time_points, phys, label=f'{label} (phys)', color=color, linewidth=2)
        # Plot phys_footprint_peak as dashed line
        ax.plot(time_points, peak, label=f'{label} (phys_peak)', color=color, linewidth=1, linestyle='--', alpha=0.7)

        # Plot neural_peak if available (dotted line)
        has_neural = any(n is not None for n in neural)
        if has_neural:
            # Fill None values with NaN for plotting
            neural_plot = [n if n is not None else float('nan') for n in neural]
            ax.plot(time_points, neural_plot, label=f'{label} (neural)', color=color, linewidth=1.5, linestyle=':', alpha=0.8)

        labels.append(label)

    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Memory (MB)', fontsize=12)
    ax.set_title('Memory Usage Over Time (Physical + Neural)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3, linestyle='-')

    # Format y-axis to show MB
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)} MB'))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Graph saved to: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot memory usage over time from footprint reports"
    )
    parser.add_argument("reports", nargs='+', type=Path, help="Memory report files")
    parser.add_argument("-o", "--output", type=Path, default="memory_plot.png",
                        help="Output image file (default: memory_plot.png)")
    parser.add_argument("-i", "--interval", type=float, default=0.5,
                        help="Sampling interval in seconds (default: 0.5)")

    args = parser.parse_args()

    for report in args.reports:
        if not report.exists():
            print(f"Error: File not found: {report}", file=sys.stderr)
            return 1

    plot_memory_reports(args.reports, args.output, args.interval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
