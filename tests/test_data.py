from shelf_vision.data import drop_from_list, read_names


def test_read_names_skips_comments_and_blanks(tmp_path):
    f = tmp_path / "list.txt"
    f.write_text("# why\n\ntrain_1.jpg\n  train_2.jpg  \n")
    assert read_names(f) == {"train_1.jpg", "train_2.jpg"}


def test_drop_from_list_matches_by_file_name():
    lines = ["./images/train_1.jpg", "./images/train_10.jpg", "", "./images/train_2.jpg"]
    assert drop_from_list(lines, {"train_1.jpg"}) == [
        "./images/train_10.jpg",
        "./images/train_2.jpg",
    ]


def test_pin_dataset_root_makes_relative_root_absolute(tmp_path):
    from shelf_vision.data import pin_dataset_root

    spec = {"path": "SKU-110K", "train": "train.txt", "names": {0: "object"}}
    assert pin_dataset_root(spec, tmp_path)["path"] == str(tmp_path / "SKU-110K")
    assert pin_dataset_root({"path": "/abs/ds"}, tmp_path)["path"] == "/abs/ds"
