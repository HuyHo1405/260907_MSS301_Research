"""Phase 6 batch ALPR benchmark. Run with python -m evaluation.run_benchmark."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .benchmark_data import load_dataset, read_json, resolve
from .benchmark_profiler import RSSSampler, summarize
from .metrics import evaluate_detection, evaluate_ocr, pair_ocr_records, predictions_from_pipeline


def write_json(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def execute_cell(bundle, weights, resolution, args, factory=None):
    import cv2
    if factory is None:
        from pipeline.alpr_pipeline import ALPRPipeline
        factory = ALPRPipeline
    predictions, samples = [], []
    with RSSSampler() as memory:
        pipeline = factory(detector_weights=str(weights), engine="paddle",
                           conf_threshold=args.detector_confidence, imgsz=resolution,
                           num_threads=args.threads, iou_threshold=args.nms_iou)
        def infer(image):
            return pipeline.predict(image, profile=True, annotate=False, fallback_ocr=False,
                                    ocr_conf_threshold=args.confidence_threshold)
        records = bundle["records"]
        for i in range(args.warmup):
            image = cv2.imread(str(records[i % len(records)]["path"]))
            if image is None:
                raise ValueError("Unreadable warmup image")
            infer(image)
        for repeat in range(args.repeats):
            for record in records:
                start = time.perf_counter()
                image = cv2.imread(str(record["path"]))
                io = (time.perf_counter()-start)*1000
                if image is None:
                    raise ValueError(f"Unreadable image: {record['path']}")
                _, result, latency = infer(image)
                if repeat == 0:
                    predictions.extend(predictions_from_pipeline(record["image_id"], result))
                samples.append(dict(image_id=record["image_id"], repeat_index=repeat, io_ms=io,
                                    **latency["stages_ms"], total_ms=latency["end_to_end_ms"]))
        del pipeline
    return predictions, samples, memory.peak


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path(__file__).with_name("benchmark_config.json"))
    for name, default in (("sources", "source_a,source_b"), ("models", "FP32,INT8"),
                          ("resolutions", "640,480,320"), ("corruptions", "clean"), ("severities", "1,2,3")):
        p.add_argument("--"+name, default=default)
    for name, default in (("threads", 4), ("warmup", 20), ("repeats", 3), ("seed", 42)):
        p.add_argument("--"+name, type=int, default=default)
    p.add_argument("--sample-size", type=int)
    p.add_argument("--detector-confidence", type=float, default=.001)
    p.add_argument("--confidence-threshold", type=float, default=.25)
    p.add_argument("--nms-iou", type=float, default=.45)
    p.add_argument("--profile", default="Native CPU")
    p.add_argument("--output", type=Path, required=True)
    for flag in ("preflight-only", "allow-pseudo-labels", "overwrite"):
        p.add_argument("--"+flag, action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        print("Output exists; choose a new directory or --overwrite", file=sys.stderr)
        return 2
    output.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for name in ("benchmark.json", "samples.csv", "runs.csv", "manifest.json",
                     "preflight.json", "failures.json", "checkpoint.partial.json"):
            (output/name).unlink(missing_ok=True)
    preflight = {"errors": [], "datasets": [], "warnings": []}
    try:
        if args.warmup < 0 or min(args.repeats, args.threads) < 1 or (args.sample_size is not None and args.sample_size < 1):
            raise ValueError("Invalid warmup/repeats/threads/sample-size")
        if not 0 <= args.detector_confidence <= args.confidence_threshold <= 1 or not 0 < args.nms_iou <= 1:
            raise ValueError("Invalid confidence/NMS thresholds")
        config = read_json(args.config)
        root = (args.config.resolve().parent / config["project_root"]).resolve()
        sources, models = args.sources.split(","), args.models.split(",")
        resolutions = [int(r) for r in args.resolutions.split(",")]
        for values in (sources, models, resolutions):
            if not values or len(values) != len(set(values)):
                raise ValueError("Matrix selections must be nonempty and unique")
        if not set(sources) <= set(config["datasets"]) or not set(models) <= {"FP32", "INT8"}:
            raise ValueError("Unknown source/model")
        if not set(resolutions) <= {640, 480, 320}:
            raise ValueError("Supported resolutions: 640,480,320")
        severities = [int(s) for s in args.severities.split(",")]
        if not set(severities) <= {1, 2, 3}:
            raise ValueError("Severity must be 1,2,3")
        bundles = []
        for source in sources:
            bundle = load_dataset(root, config, source, args.sample_size, args.seed)
            bundles.append((source, "clean", 0, bundle))
        if args.corruptions != "clean":
            if "source_a" not in sources:
                raise ValueError("Corruptions require source_a")
            manifest = read_json(resolve(root, config["corruption_manifest"]))
            requested = args.corruptions.split(",")
            variants = [v for v in manifest["datasets"] if
                        (args.corruptions == "all" or v["scenario"] in requested) and v["severity"] in severities]
            scenarios = {v["scenario"] for v in variants}
            if not variants or (args.corruptions != "all" and set(requested) != scenarios):
                raise ValueError("Requested corruption is missing")
            for scenario in scenarios:
                if {v["severity"] for v in variants if v["scenario"] == scenario} != set(severities):
                    raise ValueError("Missing requested severity")
            for v in variants:
                b = load_dataset(root, config, "source_a", args.sample_size, args.seed, v)
                clean = next(b for s,c,l,b in bundles if s=="source_a" and c=="clean")
                if {g["plate_id"] for g in b["ground_truth"]} != {g["plate_id"] for g in clean["ground_truth"]} or {
                    r["image_id"] for r in b["records"]} != {r["image_id"] for r in clean["records"]}:
                    raise ValueError("Clean/corrupted cohorts differ; supply matching complete manifests")
                bundles.append(("source_a", v["scenario"], v["severity"], b))
        for source, corruption, severity, b in bundles:
            preflight["datasets"].append(dict(source=source, corruption=corruption, severity=severity,
                images=len(b["records"]), plates=len(b["ground_truth"]), unknown_ocr=len(b["invalid_ocr"]),
                labels_verified=b["labels_verified"], cohort_id=b["cohort_id"]))
            if b["invalid_ocr"]:
                preflight["errors"].append(f"{source}/{corruption}: {len(b['invalid_ocr'])} invalid OCR labels")
            if not b["labels_verified"] and not args.allow_pseudo_labels:
                preflight["errors"].append(f"{source}: labels_verified=false; exploratory runs require --allow-pseudo-labels")
        names = [{r["path"].name for r in b["records"]} for _,c,_,b in bundles if c=="clean"]
        if len(names)>1:
            preflight["warnings"].append({"shared_filenames":len(set.intersection(*names))})
        for model in models:
            if not resolve(root, config["models"][model]).is_file():
                preflight["errors"].append(f"Missing weights: {model}")
        preflight["matrix"] = [
            dict(model=m, resolution=r, source=s, corruption=c, severity=l)
            for s,c,l,b in bundles for m in models for r in resolutions]
    except (ValueError, KeyError, OSError, TypeError) as error:
        preflight["errors"].append(str(error))
    write_json(output/"preflight.json", preflight)
    if preflight["errors"] or args.preflight_only:
        return 2 if preflight["errors"] else 0
    manifest = dict(complete=False, config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                    started_at=datetime.now(timezone.utc).isoformat(), environment=dict(
                        platform=platform.platform(), machine=platform.machine(), python=sys.version,
                        packages={d.metadata["Name"]:d.version for d in importlib.metadata.distributions()}),
                    preflight=preflight, cells=[])
    runs, all_samples, summaries = [], [], []
    write_json(output/"manifest.json", manifest)
    try:
        for source, corruption, severity, bundle in bundles:
            for model in models:
                weights = resolve(root, config["models"][model])
                for resolution in resolutions:
                    run_id = f"{model}-{source}-{resolution}-{corruption}-{severity}"
                    print(f"Running {run_id}", flush=True)
                    preds, samples, peak = execute_cell(bundle, weights, resolution, args)
                    metrics = dict(detection=evaluate_detection(bundle["ground_truth"], preds,
                                   confidence_threshold=args.confidence_threshold),
                                   ocr=evaluate_ocr(pair_ocr_records(bundle["ground_truth"], preds,
                                   confidence_threshold=args.confidence_threshold)))
                    runs.append(dict(run_id=run_id, model=model, source=source, resolution=resolution,
                        profile=args.profile, corruption=corruption, severity=severity,
                        confidence_threshold=args.confidence_threshold, ground_truth=bundle["ground_truth"],
                        predictions=preds, images={r["image_id"]:r["path"].relative_to(root).as_posix() for r in bundle["records"]},
                        latency_ms=[s["total_ms"] for s in samples], peak_ram_mb=peak,
                        metadata=dict(cohort_id=bundle["cohort_id"], ram_unit="MiB", ram_method="sampled_process_rss",
                            ram_interval_ms=10, ram_scope="reused_process_including_load_and_warmup",
                            model_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                            annotation_sha256=bundle["annotation_sha256"],
                            quality_status="verified" if bundle["labels_verified"] else "exploratory"),
                        metrics=metrics))
                    all_samples.extend(dict(run_id=run_id, **s) for s in samples)
                    summaries.append(dict(run_id=run_id, images=len(bundle["records"]),
                                          model=model, resolution=resolution, source=source,
                                          corruption=corruption, severity=severity,
                                          map50=metrics["detection"]["map50"],
                                          precision=metrics["detection"]["precision"],
                                          recall=metrics["detection"]["recall"],
                                          em_percent=metrics["ocr"]["overall"]["em_percent"],
                                          cer=metrics["ocr"]["overall"]["cer"],
                                          peak_ram_mb=peak, **summarize([s["total_ms"] for s in samples])))
                    manifest["cells"].append(dict(run_id=run_id, status="complete"))
                    write_json(output/"manifest.json", manifest)
        write_csv(output/"samples.csv", all_samples)
        write_csv(output/"runs.csv", summaries)
        write_json(output/"benchmark.json", dict(schema_version=1, runs=runs))
        manifest["complete"] = True
        write_json(output/"manifest.json", manifest)
        return 0
    except Exception as error:
        write_json(output/"failures.json", dict(error=str(error), run_id=run_id))
        write_json(output/"checkpoint.partial.json", dict(complete=False, schema_version=1, runs=runs))
        manifest["error"] = str(error)
        write_json(output/"manifest.json", manifest)
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
