import detection


def test_merge_detections():
    lidar_feats = [{"bbox": (0, 0, 10, 10), "lidar": True}]
    sat_feats = [{"bbox": (5, 0, 10, 10), "sat": True}]

    merged = detection.merge_detections(lidar_feats, sat_feats, threshold=10)

    assert len(merged) == 1
    feat = merged[0]
    assert feat["sources"] == ["lidar", "satellite"]
    assert feat["bbox"] == (0, 0, 15, 10)

