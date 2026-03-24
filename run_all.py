"""
Master script to run the complete experiment pipeline.
Run this to reproduce all results for the paper.

Usage:
    python run_all.py              # run everything
    python run_all.py --skip-ablation  # skip ablation (slow with quantum)
    python run_all.py --comparison-only # just regenerate comparison figures
"""

import sys
import time


def run_step(description, module_name):
    print(f"\n{'#' * 70}")
    print(f"# {description}")
    print(f"{'#' * 70}\n")

    t0 = time.time()
    mod = __import__(module_name)
    mod.main()
    elapsed = time.time() - t0
    print(f"\n  >> {description} completed in {elapsed/60:.1f} minutes\n")


def main():
    args = set(sys.argv[1:])

    if "--comparison-only" in args:
        run_step("Step 4: Generating comparison figures and tables", "run_comparison")
        return

    run_step("Step 1: Classical DLinear Benchmark", "run_classical")

    run_step("Step 2: Quantum ADQRL", "run_quantum")

    run_step("Step 3: Novel MSQD Model", "run_msqd")

    if "--skip-ablation" not in args:
        run_step("Step 4: Ablation Study", "run_ablation")

    run_step("Step 5: Generating comparison figures and tables", "run_comparison")

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)
    print("\nOutputs saved to:")
    print(f"  outputs/classical_dlinear/   — DLinear benchmark")
    print(f"  outputs/quantum_adqrl/       — ADQRL results")
    print(f"  outputs/msqd/                — MSQD results")
    print(f"  outputs/ablation/            — Ablation study")
    print(f"  outputs/comparison/          — Cross-model comparisons & LaTeX tables")
    print(f"\nAll runs are also tracked on Weights & Biases.")


if __name__ == "__main__":
    main()
