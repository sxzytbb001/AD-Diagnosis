#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""AD diagnosis project unified entry point.

Default pipeline:
    feature selection -> Transformer training -> SVM training -> external validation

Optional analysis steps:
    gene expression boxplots
    Transformer gene interaction analysis
"""

import argparse
import sys
import time
import traceback

from . import config


def _safe_print(message=""):
    """Print safely on Windows terminals with limited encodings."""
    try:
        print(message)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        safe_message = str(message).encode(encoding, errors="replace").decode(encoding, errors="replace")
        print(safe_message)


def _run_step(step_name, step_func, enabled=True):
    if not enabled:
        _safe_print(f"[SKIP] {step_name}")
        return {"name": step_name, "status": "skipped", "seconds": 0.0}

    _safe_print(f"\n>>> Start: {step_name}")
    start_time = time.time()
    try:
        step_func()
        elapsed = time.time() - start_time
        _safe_print(f"[OK] {step_name} ({elapsed:.1f}s)")
        return {"name": step_name, "status": "success", "seconds": elapsed}
    except Exception as exc:
        elapsed = time.time() - start_time
        _safe_print(f"[ERR] {step_name} ({elapsed:.1f}s)")
        _safe_print(f"Reason: {exc}")
        traceback.print_exc()
        return {"name": step_name, "status": "failed", "seconds": elapsed, "error": str(exc)}


def run_full_pipeline(
    skip_feature_selection=False,
    skip_transformer=False,
    skip_svm=False,
    skip_external_validation=False,
    with_gene_expression=False,
    with_gene_interaction=False,
    continue_on_error=False,
):
    """Run the complete AD diagnosis workflow."""
    config.ensure_dirs()

    _safe_print("=" * 72)
    _safe_print("AD Diagnosis Pipeline")
    _safe_print("=" * 72)
    _safe_print(f"Project root: {config.BASE_DIR}")
    _safe_print(f"Training data: {config.TRAIN_DATA_DIR}")
    _safe_print(f"Results dir: {config.RESULTS_DIR}")
    if with_gene_expression or with_gene_interaction:
        _safe_print("Optional analyses enabled:")
        if with_gene_expression:
            _safe_print("  - Gene expression boxplots")
        if with_gene_interaction:
            _safe_print("  - Transformer gene interaction analysis")

    from .feature_selection import feature_selection
    from .train_transformer import train_transformer
    from .train_svm import train_svm
    from .external_validation import evaluate_external_models

    steps = [
        ("Feature selection", feature_selection, not skip_feature_selection),
        ("Transformer training", train_transformer, not skip_transformer),
        ("SVM training", train_svm, not skip_svm),
        ("External validation", evaluate_external_models, not skip_external_validation),
    ]

    if with_gene_expression:
        from .plot_gene_expression import main as plot_gene_expression_main

        steps.append(("Gene expression boxplots", plot_gene_expression_main, True))

    if with_gene_interaction:
        from .plot_gene_interaction import main as plot_gene_interaction_main

        steps.append(("Transformer gene interaction analysis", plot_gene_interaction_main, True))

    results = []
    for step_name, step_func, enabled in steps:
        outcome = _run_step(step_name, step_func, enabled=enabled)
        results.append(outcome)

        if outcome["status"] == "failed" and not continue_on_error:
            _safe_print("\nPipeline stopped because a step failed.")
            break

    succeeded = sum(item["status"] == "success" for item in results)
    failed = [item for item in results if item["status"] == "failed"]
    skipped = sum(item["status"] == "skipped" for item in results)

    _safe_print("\n" + "=" * 72)
    _safe_print("Pipeline summary")
    _safe_print("=" * 72)
    for item in results:
        status = item["status"]
        icon = "[OK]" if status == "success" else "[ERR]" if status == "failed" else "[SKIP]"
        _safe_print(f"{icon} {item['name']} - {status}")

    _safe_print("-" * 72)
    _safe_print(f"Success: {succeeded} | Failed: {len(failed)} | Skipped: {skipped}")
    _safe_print(f"Results dir: {config.RESULTS_DIR}")
    _safe_print("=" * 72)

    return len(failed) == 0


def main():
    parser = argparse.ArgumentParser(description="Run the AD diagnosis pipeline.")
    parser.add_argument("--skip-feature-selection", action="store_true", help="Skip feature selection.")
    parser.add_argument("--skip-transformer", action="store_true", help="Skip Transformer training.")
    parser.add_argument("--skip-svm", action="store_true", help="Skip SVM training.")
    parser.add_argument("--skip-external-validation", action="store_true", help="Skip external validation.")
    parser.add_argument(
        "--with-gene-expression",
        action="store_true",
        help="Generate gene expression boxplots as an optional post-hoc analysis.",
    )
    parser.add_argument(
        "--with-gene-interaction",
        action="store_true",
        help="Generate Transformer gene interaction figures as an optional post-hoc analysis.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue the remaining steps after a failure.",
    )
    args = parser.parse_args()

    ok = run_full_pipeline(
        skip_feature_selection=args.skip_feature_selection,
        skip_transformer=args.skip_transformer,
        skip_svm=args.skip_svm,
        skip_external_validation=args.skip_external_validation,
        with_gene_expression=args.with_gene_expression,
        with_gene_interaction=args.with_gene_interaction,
        continue_on_error=args.continue_on_error,
    )

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
