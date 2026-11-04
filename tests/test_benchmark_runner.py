import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.run_benchmark import main
from evaluation.benchmark_data import load_dataset
from evaluation.benchmark_profiler import summarize
from evaluation.visualizer import load_runs


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root/"images").mkdir()
        for name in ("a.jpg", "negative.jpg"):
            (self.root/"images"/name).write_bytes(b"fixture")
        gt = [dict(file_name="a.jpg", ann_id=i, bbox=[i, 0, i+1, 1],
                   width=10, height=10, text="30A12345", class_id=1, layout="one_line") for i in (1, 2)]
        (self.root/"gt.json").write_text(json.dumps(gt))
        coco = dict(images=[dict(id=i, file_name=n) for i,n in enumerate(("a.jpg","negative.jpg"))],
                    annotations=[dict(id=i, image_id=0) for i in (1,2)])
        (self.root/"coco.json").write_text(json.dumps(coco))
        for name in ("fp.onnx","int.onnx"):
            (self.root/name).write_bytes(b"weights")
        self.config = dict(project_root=".", models=dict(FP32="fp.onnx",INT8="int.onnx"),
            datasets=dict(source_a=dict(images="images",annotations="coco.json",
                                       ground_truth="gt.json",labels_verified=True)))
        self.path = self.root/"config.json"
        self.path.write_text(json.dumps(self.config))
        self.output = self.root/"out"

    def args(self):
        return ["--config",str(self.path),"--sources","source_a","--output",str(self.output)]

    def test_multi_plate_and_negative_inventory(self):
        b = load_dataset(self.root,self.config,"source_a")
        self.assertEqual(len(b["records"]),2)
        self.assertEqual(len(b["ground_truth"]),2)
        self.assertEqual(len({g["plate_id"] for g in b["ground_truth"]}),2)

    def test_unknown_labels_block_before_inference(self):
        gt = json.loads((self.root/"gt.json").read_text())
        gt[0]["text"]="UNKNOWN"
        (self.root/"gt.json").write_text(json.dumps(gt))
        with patch("evaluation.run_benchmark.execute_cell") as execute:
            self.assertEqual(main(self.args()+["--allow-pseudo-labels"]),2)
            execute.assert_not_called()

    def test_duplicate_matrix_rejected(self):
        with patch("evaluation.run_benchmark.execute_cell") as execute:
            self.assertEqual(main(self.args()+["--resolutions","480,480"]),2)
            execute.assert_not_called()

    def test_matrix_and_report_contract(self):
        def execute(bundle, weights, resolution, args):
            predictions = [dict(g, score=.9) for g in bundle["ground_truth"]]
            samples = [dict(image_id=r["image_id"],repeat_index=i,io_ms=1,preprocess_ms=1,
                            detection_ms=2,postprocess_ms=1,crop_roi_ms=1,ocr_ms=2,
                            overhead_ms=1,total_ms=8) for i in range(args.repeats) for r in bundle["records"]]
            return predictions,samples,30.0
        with patch("evaluation.run_benchmark.execute_cell",side_effect=execute):
            self.assertEqual(main(self.args()+["--repeats","2"]),0)
        runs=load_runs(self.output/"benchmark.json")
        self.assertEqual(len(runs),6)
        self.assertEqual(len(runs[0]["predictions"]),2)
        self.assertEqual(len(runs[0]["latency_ms"]),4)
        self.assertEqual(runs[0]["metrics"]["ocr"]["overall"]["em_percent"],100)
        self.assertTrue(json.loads((self.output/"manifest.json").read_text())["complete"])

    def test_missing_lineage_rejected(self):
        v=dict(image_dir="images",annotations_file="coco.json")
        with self.assertRaisesRegex(ValueError,"original_annotation_id"):
            load_dataset(self.root,self.config,"source_a",variant=v)

    def test_path_escape_rejected(self):
        self.config["datasets"]["source_a"]["images"]="../outside"
        with self.assertRaisesRegex(ValueError,"outside"):
            load_dataset(self.root,self.config,"source_a")

    def test_worker_error_does_not_publish_success(self):
        with patch("evaluation.run_benchmark.execute_cell",side_effect=RuntimeError("model error")):
            self.assertEqual(main(self.args()),1)
        self.assertFalse((self.output/"benchmark.json").exists())
        self.assertFalse(json.loads((self.output/"manifest.json").read_text())["complete"])

    def test_percentiles(self):
        s=summarize([10,20])
        self.assertEqual(s["mean"],15)
        self.assertEqual(s["p95"],19.5)
        self.assertEqual(s["std"],5)


if __name__ == "__main__":
    unittest.main()
