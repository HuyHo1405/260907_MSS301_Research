import unittest
from unittest.mock import Mock
import numpy as np
from pipeline.alpr_pipeline import ALPRPipeline


class ProfilingTests(unittest.TestCase):
    def pipeline(self, boxes):
        p=ALPRPipeline.__new__(ALPRPipeline)
        p.detector=Mock()
        p.detector.last_profile=dict(preprocess_ms=0,detection_ms=0,postprocess_ms=0,crop_roi_ms=0)
        p.detector.detect.return_value=boxes
        p.recognizer=Mock()
        p.recognizer.recognize.return_value=(["ABC"],"ABC",0.01)
        return p

    def test_no_detection_no_fallback_and_no_drawing(self):
        p=self.pipeline(([],[],[],0))
        img=np.zeros((10,10,3),dtype=np.uint8)
        out,results,timing=p.predict(img,profile=True,annotate=False,fallback_ocr=False)
        self.assertIs(out,img)
        self.assertEqual(results,[])
        p.recognizer.recognize.assert_not_called()
        self.assertEqual(timing["stages_ms"]["ocr_ms"],0)
        self.assertGreater(timing["end_to_end_ms"],0)

    def test_low_confidence_retained_without_ocr(self):
        img=np.zeros((10,10,3),dtype=np.uint8)
        p=self.pipeline(([[0,0,5,5]],[.1],[img],.1))
        _,results,_=p.predict(img,annotate=False,fallback_ocr=False,ocr_conf_threshold=.25)
        self.assertEqual(len(results),1)
        self.assertEqual(results[0]["formatted_plate"],"")
        p.recognizer.recognize.assert_not_called()

    def test_legacy_tuple_and_latency(self):
        img=np.zeros((10,10,3),dtype=np.uint8)
        p=self.pipeline(([],[],[],.1))
        out,results,timing=p.predict(img)
        self.assertEqual(len(results),1)
        self.assertEqual(set(timing),{"det_time_ms","ocr_time_ms","total_time_ms","fps"})
        p.recognizer.recognize.assert_called_once()

    def test_worker_excludes_warmup_and_keeps_first_predictions(self):
        import argparse
        import tempfile
        from pathlib import Path
        import cv2
        from evaluation.run_benchmark import execute_cell
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"image.png"
            self.assertTrue(cv2.imwrite(str(path), np.zeros((10,10,3),dtype=np.uint8)))
            p = self.pipeline(([],[],[],0))
            factory = Mock(return_value=p)
            args = argparse.Namespace(detector_confidence=.001, threads=2,nms_iou=.45,
                                      confidence_threshold=.25,warmup=2,repeats=3)
            bundle = dict(records=[dict(image_id="one",path=path)])
            predictions,samples,peak = execute_cell(bundle,path,480,args,factory)
            self.assertEqual(p.detector.detect.call_count,5)
            self.assertEqual(len(samples),3)
            self.assertEqual(predictions,[])
            self.assertGreater(peak,0)
            self.assertEqual(factory.call_args.kwargs["imgsz"],480)
            self.assertEqual(factory.call_args.kwargs["num_threads"],2)
