import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np


UTILS_PATH = Path(__file__).resolve().parents[1] / "sarah" / "data" / "utils.py"
SPEC = importlib.util.spec_from_file_location("handwritten_text_utils", UTILS_PATH)
utils = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(utils)


class ReadImageBoundingBoxTest(unittest.TestCase):
    def test_bbox_near_top_left_does_not_reflect_crop_inward(self):
        image = np.full((32, 32), 255, dtype=np.uint8)
        image[3:6, 4:7] = 0

        with patch.object(utils.cv2, "imread", return_value=image):
            cropped = utils.read_image("synthetic.png", ["4", "3", "3", "3"])

        self.assertEqual(9, int(np.count_nonzero(cropped == 0)))

    def test_bbox_with_left_margin_preserves_requested_pixels(self):
        image = np.full((32, 32), 255, dtype=np.uint8)
        image[13:16, 14:17] = 0

        with patch.object(utils.cv2, "imread", return_value=image):
            cropped = utils.read_image("synthetic.png", ["14", "13", "3", "3"])

        self.assertEqual(9, int(np.count_nonzero(cropped == 0)))


if __name__ == "__main__":
    unittest.main()
