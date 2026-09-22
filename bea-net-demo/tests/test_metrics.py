"""Small deterministic tests; no upstream checkout or checkpoint is required."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_demo import metrics


class MetricTests(unittest.TestCase):
    def test_perfect(self):
        self.assertEqual(metrics([1, 0], [1, 0])['dice'], 1.0)

    def test_disjoint(self):
        self.assertEqual(metrics([1, 0], [0, 1])['dice'], 0.0)

    def test_partial(self):
        result = metrics([1, 1, 0], [1, 0, 1])
        for key in ('dice', 'precision', 'recall'):
            self.assertEqual(result[key], 0.5)
        self.assertEqual(result['false_positive_pixels'], 1)
        self.assertEqual(result['false_negative_pixels'], 1)

    def test_both_empty(self):
        result = metrics([0, 0], [0, 0])
        self.assertEqual(result['dice'], 1.0)
        self.assertIsNone(result['precision'])
        self.assertIsNone(result['recall'])

    def test_no_prediction(self):
        result = metrics([0, 0], [1, 0])
        self.assertEqual(result['dice'], 0.0)
        self.assertIsNone(result['precision'])
        self.assertEqual(result['recall'], 0.0)

    def test_no_target(self):
        result = metrics([1, 0], [0, 0])
        self.assertEqual(result['dice'], 0.0)
        self.assertEqual(result['precision'], 0.0)
        self.assertIsNone(result['recall'])

    def test_shape_mismatch(self):
        with self.assertRaises(ValueError):
            metrics([1], [1, 0])


if __name__ == '__main__':
    unittest.main()
