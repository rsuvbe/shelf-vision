import numpy as np

from shelf_vision.gaps import find_gaps, rows


def shelf(xs, y=100, w=10, h=30):
    return np.array([[x, y - h, x + w, y] for x in xs], float)


def test_rows_split_by_bottom_edge():
    boxes = np.vstack([shelf([0, 12, 24]), shelf([0, 12], y=200)])
    assert sorted(len(r) for r in rows(boxes)) == [2, 3]


def test_missing_product_inside_a_row_is_a_gap():
    boxes = shelf([0, 12, 24, 48, 60])  # one product missing at x=36
    gaps = find_gaps(boxes)
    assert len(gaps) == 1 and 33 <= gaps[0, 0] <= 35 and gaps[0, 2] == 48


def test_tight_row_and_row_ends_have_no_gaps():
    assert len(find_gaps(shelf([0, 12, 24, 36]))) == 0
    assert len(find_gaps(shelf([100, 112, 124]))) == 0  # empty space left of the row is ignored
