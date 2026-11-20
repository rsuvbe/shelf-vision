import numpy as np

from shelf_vision.pilot import PhotoScore, from_yolo, match_count, summarise, to_yolo


def test_yolo_round_trip():
    boxes = np.array([[10, 20, 50, 80], [0, 0, 100, 40]], np.float32)
    back = from_yolo(to_yolo(boxes, 200, 100), 200, 100)
    assert np.allclose(back, boxes, atol=1e-3)
    assert from_yolo("", 10, 10).shape == (0, 4) and to_yolo(np.zeros((0, 4)), 1, 1) == ""


def test_match_count_is_one_to_one():
    gt = np.array([[0, 0, 10, 10], [20, 0, 30, 10]], float)
    pred = np.array([[0, 0, 10, 10], [1, 0, 11, 10], [50, 50, 60, 60]], float)
    assert match_count(pred, gt) == 1  # the near-duplicate cannot take the same label


def test_summarise_reports_error_and_recall():
    text = summarise([PhotoScore("a", 100, 90, 85), PhotoScore("b", 50, 55, 50)])
    assert "count error per photo: 7.5 products (10.0%)" in text
    assert "recall 0.900" in text
