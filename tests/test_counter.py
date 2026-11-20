import numpy as np

from shelf_vision.counter import Letterbox, decode, letterbox, nms


def test_nms_drops_overlaps_above_threshold_only():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 9], [0, 0, 10, 5], [20, 20, 30, 30]], float)
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    # IoU with box 0: 0.9 (dropped at 0.8), 0.5 (kept), 0 (kept)
    assert nms(boxes, scores, iou=0.8, max_det=10).tolist() == [0, 2, 3]


def test_nms_orders_by_score_and_caps():
    boxes = np.array([[0, 0, 1, 1], [5, 5, 6, 6], [9, 9, 10, 10]], float)
    scores = np.array([0.2, 0.9, 0.5])
    assert nms(boxes, scores, iou=0.8, max_det=2).tolist() == [1, 2]


def test_letterbox_pads_the_short_side_and_decode_maps_back():
    img = np.zeros((100, 200, 3), np.uint8)  # wide photo
    x, lb = letterbox(img, 400)
    assert x.shape == (1, 3, 400, 400) and lb == Letterbox(2.0, 0, 100, 200, 100)
    # one box covering the photo's left half, in model-input pixels
    out = np.array([[100.0], [200.0], [200.0], [200.0], [0.9]])
    boxes, scores = decode(out, lb, conf=0.5)
    assert np.allclose(boxes, [[0, 0, 100, 100]]) and np.allclose(scores, [0.9])
    assert decode(out, lb, conf=0.95)[0].shape == (0, 4)
