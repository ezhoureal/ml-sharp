import argparse
import subprocess
import time
import os
import signal
import sys
import threading
from pathlib import Path

import numpy as np

from run_coreml_inference import load_coreml_model, preprocess_image, run_inference


def run_model_inference(model_path: Path, image_path: Path) -> None:
    """Run Core ML model inference for profiling.

    Args:
        model_path: Path to the Core ML model (.mlpackage file).
        image_path: Path to the input image.
    """
    print("[Model] Starting model loading and inference...")

    # Load model
    print(f"[Model] Loading model from {model_path}")
    model = load_coreml_model(model_path)

    # Load image (simple numpy load for RGB)
    from sharp.utils import io

    print(f"[Model] Loading image from {image_path}")
    image, _, f_px = io.load_rgb(image_path)

    # Run inference
    print("[Model] Running inference...")
    predictions = run_inference(model, image, f_px)

    print(f"[Model] Inference complete. Outputs: {list(predictions.keys())}")

# --- PROFILER LOGIC ---
def run_benchmark(model_path: Path, image_path: Path, output_file="memory_report.txt"):
    # We need the PID of the current process to profile ourselves
    pid = os.getpid()
    
    print(f"[Profiler] Target PID: {pid}")
    print(f"[Profiler] Reports will be saved to: {output_file}")

    # 1. Prepare the footprint command
    # We use -v for verbose to get IOKit/ANE breakdown
    cmd = [
        "sudo", "footprint", 
        "-p", str(pid), 
        "--sample", "0.5",           # Sample every 0.5s
        "--sample-duration", "60",   # Max duration (failsafe)
        "-v"
    ]

    # 2. Start the model in a separate thread
    model_thread = threading.Thread(target=run_model_inference, args=(model_path, image_path))
    
    try:
        # Start the footprint process (it will ask for sudo password in terminal)
        # We redirect output to a file directly
        with open(output_file, "w") as f:
            monitor_proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.PIPE)
            
            print("[Profiler] Monitor started. Please enter sudo password if prompted.")
            time.sleep(2) # Give monitor a moment to initialize
            
            start_time = time.time()
            model_thread.start()
            
            # Wait for the model to finish
            model_thread.join()
            end_time = time.time()

        # 3. Clean up the monitor
        monitor_proc.terminate()
        print(f"\n[Profiler] Benchmark finished in {end_time - start_time:.2f}s")
        print("-" * 40)
        analyze_report(output_file)

    except Exception as e:
        print(f"Error during profiling: {e}")

def analyze_report(file_path):
    """Quickly scans the footprint file for the most important 'Unified' metrics."""
    try:
        with open(file_path, "r") as f:
            content = f.read()
            
        print("QUICK SUMMARY FROM FOOTPRINT:")
        # Look for the physical footprint (The true cost to the OS)
        for line in content.splitlines():
            if "phys_footprint" in line:
                print(f"  > {line.strip()}")
                
    except FileNotFoundError:
        print("Could not find the report file for analysis.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile Core ML model inference using footprint")
    parser.add_argument(
        "-m", "--model-path",
        type=Path,
        required=True,
        help="Path to the Core ML model (.mlpackage file)"
    )
    parser.add_argument(
        "-i", "--image-path",
        type=Path,
        default="data/sample.jpg",
        help="Path to the input image"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="memory_report.txt",
        help="Output file for footprint report (default: memory_report.txt)"
    )
    args = parser.parse_args()

    if os.geteuid() != 0:
        print("NOTE: This script works best when run with 'sudo uv run python auto_profile.py'")
        print("to avoid multiple password prompts.")

    run_benchmark(args.model_path, args.image_path, args.output)