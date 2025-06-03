import numpy as np
import pytest

import agents.eyes_tools as eyes_tools


def test_detect_image_features_runtime(tmp_path):
    img_path = tmp_path / "img.png"
    if eyes_tools.cv2 is None or eyes_tools.np is None:
        with pytest.raises(RuntimeError):
            eyes_tools.detect_image_features(str(img_path))
    else:
        img = np.zeros((50, 50), dtype=np.uint8)
        eyes_tools.cv2.rectangle(img, (10, 10), (40, 40), 255, -1)
        eyes_tools.cv2.imwrite(str(img_path), img)
        result = eyes_tools.detect_image_features(str(img_path))
        assert "num_contours" in result
        assert "features" in result
        assert result["num_contours"] == len(result["features"])
        if result["features"]:
            feat = result["features"][0]
            assert set(["bbox", "width", "height", "area"]).issubset(feat)
