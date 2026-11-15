from pathlib import Path

from shelf_vision.env import Runtime


def rt(device: str, vram: float) -> Runtime:
    return Runtime("local", Path("d"), Path("r"), device, "gpu", vram, 4)


def test_batch_matches_reference_on_8gb():
    assert rt("0", 8.0).batch_for(6) == 6


def test_batch_scales_with_vram_and_gpu_count():
    # Kaggle 2x T4: 15 GB each
    assert rt("0,1", 15.0).batch_for(6) == 22


def test_batch_on_cpu_is_small_but_positive():
    assert rt("cpu", 0.0).batch_for(2) == 1
