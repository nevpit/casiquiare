import numpy as np
import pytest

import agents.eyes.tools as eyes_tools


def test_save_snippets_runtime(tmp_path):
    img = np.zeros((256, 256), dtype=np.uint8)
    features = [{"bbox": (100, 100, 20, 20)}]
    if (
        eyes_tools.cv2 is None
        or eyes_tools.np is None
        or eyes_tools.to_uint8 is None
    ):
        with pytest.raises(RuntimeError):
            eyes_tools.save_snippets(img, features, str(tmp_path))
    else:
        paths = eyes_tools.save_snippets(img, features, str(tmp_path))
        assert len(paths) == 1
        assert paths[0].endswith(".png")
        assert (tmp_path / paths[0].split("/")[-1]).exists()

