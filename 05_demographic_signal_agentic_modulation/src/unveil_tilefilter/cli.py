"""Command line: ``unveil-tilefilter <step> ...`` (or ``python -m unveil_tilefilter``).

    signal    per-tile demographic signal from your demographic classifier
    masks     keep-masks from the signals (the filter)
    fairtune  train with the collaborators' fairtune_pipeline, training bags filtered
              (or, with --inference, score a trained model with nothing filtered)
    resolve   score finished runs with the resolution rule

Every option has a default or is a path you give; nothing is fixed. ``<step> --help``
lists them.
"""

import argparse
import csv
import json
import os
import sys

from . import __version__
from .filters import FILTERS, FilterParams


def _read_list(path):
    """Slide ids, one per line (blank lines and # comments ignored)."""
    if path is None:
        return None
    with open(path) as fh:
        return [ln.strip() for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]


def _params(a):
    return FilterParams(drop_frac=a.drop_frac, evidence_base=a.evidence_base,
                        attention_gain=a.attention_gain, attention_bonus=a.attention_bonus,
                        decision_threshold=a.decision_threshold, min_tiles=a.min_tiles,
                        min_keep_frac=a.min_keep_frac).validate()


def cmd_signal(a):
    from .signal import compute_signals

    m = compute_signals(a.features_dir, a.checkpoint, a.out, code_dirs=a.model_code,
                        checkpoint_key=a.checkpoint_key, adapter=a.adapter,
                        model_config=a.model_config,
                        feature_ext=a.feature_ext, h5_key=a.h5_key,
                        slides=_read_list(a.slides), chunk=a.chunk, threads=a.threads,
                        dtype=a.dtype, activation=a.activation, overwrite=a.overwrite)
    return 1 if m["slides_failed"] else 0


def cmd_masks(a):
    from .masks import build_masks, write_mask_file

    p = _params(a)
    masks = build_masks(a.signals, a.filter, p, a.seed, _read_list(a.slides),
                        a.check_features_dir, a.feature_ext, a.h5_key)
    meta = write_mask_file(a.out, masks, a.filter, p, a.seed, a.signals)
    print(json.dumps({"out": a.out, "mask_id": meta["mask_id"], **meta["stats"]}, indent=1))
    return 0


def cmd_fairtune(a, passthrough):
    from .fairtune import AdapterError, run

    try:
        run(a.upstream, a.entrypoint, passthrough, masks=a.masks, overlay=a.overlay,
            entrypoint_script=a.entrypoint_script, record=a.record,
            allow_upstream_drift=a.allow_upstream_drift, seed_numpy=a.seed_numpy,
            check_only=a.check_only, inference=a.inference)
    except AdapterError as e:
        print(f"[fairtune] STOPPED: {e}", file=sys.stderr)
        return 2
    return 0


def cmd_resolve(a):
    from .resolve import DEFAULT_METRICS, parse_run, resolve

    metrics = tuple(a.metrics.split(",")) if a.metrics else DEFAULT_METRICS
    rows = []
    if a.table:
        with open(a.table, newline="") as fh:
            for r in csv.DictReader(fh):
                rows.append((r.get("name") or r["run_dir"], r["run_dir"],
                             r.get("baseline_dir") or None,
                             float(r["baseline_auroc"]) if r.get("baseline_auroc") else None))
    for d in a.run_dir or []:
        rows.append((d, d, a.baseline_dir, a.baseline_auroc))
    if not rows:
        print("resolve: give --run-dir or --table", file=sys.stderr)
        return 2
    out = []
    for name, run_dir, bdir, bauc in rows:
        if bauc is None and bdir:
            b = parse_run(bdir, metrics, status_check=not a.no_status_check,
                          ignore_error_keys=a.ignore_error_keys)
            bauc = None if (b is None or b.get("status_problems")) else b.get("auroc")
        rec = parse_run(run_dir, metrics, status_check=not a.no_status_check,
                        ignore_error_keys=a.ignore_error_keys)
        r = resolve(rec, bauc, metrics, a.alpha, a.p_rule, not a.allow_missing_p,
                    a.auroc_margin, check_auroc=not a.no_auroc_check)
        r["name"] = name
        out.append(r)
    cols = (["name", "resolved", "fair", "keeps_auroc", "auroc", "baseline_auroc", "d_auroc",
             "auroc_p_value"] + [c for m in metrics for c in (m, "p_" + m)]
            + ["prefix", "why", "task_dir"])
    stream = open(a.out, "w", newline="") if a.out else sys.stdout
    w = csv.DictWriter(stream, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in out:
        w.writerow({k: r.get(k) for k in cols})
    if a.out:
        stream.close()
        print(f"resolve: {sum(r['resolved'] for r in out)}/{len(out)} resolved -> {a.out}")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    passthrough = []
    if "--" in argv:
        i = argv.index("--")
        argv, passthrough = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(prog="unveil-tilefilter", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"unveil_tilefilter {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    D = argparse.ArgumentDefaultsHelpFormatter

    s = sub.add_parser("signal", formatter_class=D,
                       help="compute the demographic signal of every slide")
    s.add_argument("--features-dir", required=True,
                   help="one feature file per slide; its stem is the slide's folder_id")
    s.add_argument("--checkpoint", required=True, help="your demographic classifier")
    s.add_argument("--out", required=True, help="signal directory to write")
    s.add_argument("--model-code", action="append", default=[],
                   help="directory with the classifier's Python code (repeatable)")
    s.add_argument("--checkpoint-key", default="model",
                   help="key of the model object in a pickled checkpoint; '' for a bare object")
    s.add_argument("--model-config", default=None,
                   help="JSON with model_type and model_init_args: the checkpoint is then a "
                        "plain state_dict (the released classifiers)")
    s.add_argument("--adapter", default="gated_abmil",
                   help="gated_abmil, or module:function returning "
                        "(attention, tile_score, slide_pred) for (model, x)")
    s.add_argument("--feature-ext", default=".pt", choices=[".pt", ".npy", ".h5"])
    s.add_argument("--h5-key", default="features")
    s.add_argument("--slides", default=None, help="file of folder_ids to process (default all)")
    s.add_argument("--chunk", type=int, default=2048, help="tiles per forward chunk")
    s.add_argument("--threads", type=int, default=1, help="torch CPU threads")
    s.add_argument("--dtype", default="float16", choices=["float16", "float32"],
                   help="storage precision of attention and tile_score")
    s.add_argument("--activation", default="none", choices=["none", "sigmoid"],
                   help="sigmoid if your model returns logits instead of probabilities")
    s.add_argument("--overwrite", action="store_true", help="recompute existing slides")

    m = sub.add_parser("masks", formatter_class=D, help="build keep-masks from the signals")
    m.add_argument("--signals", required=True, help="signal directory (step 1)")
    m.add_argument("--out", required=True, help="mask file to write (.npz; a .json sidecar "
                                                "is written next to it)")
    m.add_argument("--filter", default="adaptive", choices=FILTERS)
    m.add_argument("--seed", type=int, default=0, help="random_matched's draw (others ignore it)")
    m.add_argument("--slides", default=None, help="file of folder_ids (default all signals)")
    m.add_argument("--check-features-dir", default=None,
                   help="if given, check each slide's feature rows against its signal length")
    m.add_argument("--feature-ext", default=".pt", choices=[".pt", ".npy", ".h5"])
    m.add_argument("--h5-key", default="features")
    d = FilterParams()
    for name, typ, help_ in (
            ("drop-frac", float, "fraction of a slide's tiles to drop"),
            ("evidence-base", float, "score = evidence*(base + gain*rank) + bonus*rank"),
            ("attention-gain", float, "see --evidence-base"),
            ("attention-bonus", float, "see --evidence-base"),
            ("decision-threshold", float, "slide_pred >= this means group 1"),
            ("min-tiles", int, "slides with at most this many tiles are not filtered"),
            ("min-keep-frac", float, "keep at least max(min-tiles, ceil(this*n)) tiles")):
        m.add_argument("--" + name, type=typ, default=getattr(d, name.replace("-", "_")),
                       help=help_ + "; the default is the paper's value")

    f = sub.add_parser("fairtune", formatter_class=D,
                       help="train with fairtune_pipeline, training bags filtered, or score a "
                            "trained model (--inference); put the pipeline's own arguments "
                            "after --")
    f.add_argument("--upstream", required=True, help="the fairtune_pipeline checkout")
    f.add_argument("--entrypoint", default="mutation",
                   help="mutation (main_genetic.py), or stage/grade/classification/"
                        "detection (main.py)")
    f.add_argument("--entrypoint-script", default=None, help="run this script instead")
    f.add_argument("--masks", action="append", default=[],
                   help="mask file(s) for the training slides (repeatable); omit for an "
                        "unfiltered (baseline) run")
    f.add_argument("--inference", action="store_true",
                   help="score a trained model (external cohort, or a TCGA model's own "
                        "partitions) with the pipeline's --inference_only, or re-run a "
                        "finished run's bias test with its --sig_test_only (one of the two): "
                        "no --masks, keep the --train_method it was trained under; nothing "
                        "is filtered. By default the pipeline writes into the run's own "
                        "folder, replacing its files (README, step 3b)")
    f.add_argument("--overlay", default=None, help="overlay directory (default ./unveil_overlay)")
    f.add_argument("--record", default=None,
                   help="run record (default <model_path>/unveil_tilefilter_run_<time>_<pid>"
                        ".json)")
    f.add_argument("--allow-upstream-drift", action="store_true",
                   help="run even if the pipeline's callables differ from the tested version")
    f.add_argument("--seed-numpy", type=int, default=None,
                   help="seed the main process's numpy RNG; it draws the max_train_tiles "
                        "subsample only with the pipeline's --n_workers 0 (workers reseed "
                        "numpy from the run's torch seed)")
    f.add_argument("--check-only", action="store_true",
                   help="build the overlay and check the pipeline, then stop")

    r = sub.add_parser("resolve", formatter_class=D, help="score finished runs")
    r.add_argument("--run-dir", action="append", default=None, help="task folder (repeatable)")
    r.add_argument("--baseline-dir", default=None, help="the baseline's task folder")
    r.add_argument("--baseline-auroc", type=float, default=None, help="or its AUROC")
    r.add_argument("--table", default=None,
                   help="CSV with run_dir and baseline_dir or baseline_auroc (and name)")
    r.add_argument("--metrics", default=None,
                   help="comma-separated fairness metrics that must all be non-significant "
                        "(default EOddAbs, the study's rule; the pipeline also writes "
                        "EOpp0,EOpp1,EBAcc,AUCDiff,ACC_diff)")
    r.add_argument("--alpha", type=float, default=0.05)
    r.add_argument("--p-rule", default="ge", choices=["ge", "gt"],
                   help="ge: fair if every p >= alpha; gt: every p > alpha")
    r.add_argument("--auroc-margin", type=float, default=0.02,
                   help="keeps AUROC if AUROC - baseline >= -margin")
    r.add_argument("--allow-missing-p", action="store_true",
                   help="count a missing p-value as not significant")
    r.add_argument("--no-auroc-check", action="store_true", help="score fairness only")
    r.add_argument("--no-status-check", action="store_true",
                   help="score runs whose status record shows a failure")
    r.add_argument("--ignore-error-keys", action="store_true",
                   help="score a 'success' run despite error_class/error_msg/traceback when "
                        "both bootstrap files postdate its ended_at (the pipeline authors' "
                        "rule). It cannot tell keys left by an earlier, rerun attempt from a "
                        "fold that failed in the same attempt: use it only for runs whose job "
                        "logs you have checked")
    r.add_argument("--out", default=None, help="CSV to write (default stdout)")

    a = ap.parse_args(argv)
    if a.cmd != "fairtune" and passthrough:
        ap.error("arguments after -- are for the fairtune step only")
    if a.cmd == "signal":
        return cmd_signal(a)
    if a.cmd == "masks":
        return cmd_masks(a)
    if a.cmd == "fairtune":
        return cmd_fairtune(a, passthrough)
    return cmd_resolve(a)


if __name__ == "__main__":
    sys.exit(main())
