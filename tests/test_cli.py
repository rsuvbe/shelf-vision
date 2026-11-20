from shelf_vision.cli import infer_args


def test_infer_section_wins_over_train():
    cfg = {"train": {"imgsz": 960, "max_det": 1000}, "infer": {"imgsz": 1280, "iou": 0.8}}
    assert infer_args(cfg) == {"imgsz": 1280, "iou": 0.8, "max_det": 1000}


def test_falls_back_to_train_settings_and_default_iou():
    cfg = {"train": {"imgsz": 960}}
    assert infer_args(cfg) == {"imgsz": 960, "iou": 0.7, "max_det": 300}


def test_explicit_arguments_win():
    cfg = {"train": {"imgsz": 960}, "infer": {"imgsz": 1280, "iou": 0.8}}
    assert infer_args(cfg, imgsz=960, iou=0.7)["imgsz"] == 960
    assert infer_args(cfg, imgsz=960, iou=0.7)["iou"] == 0.7
