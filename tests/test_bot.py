import io

from PIL import Image

from shelf_vision.bot import caption, parse_allowed, parse_conf
from shelf_vision.counter import decode_image


def test_parse_conf_accepts_comma_and_range():
    assert parse_conf(["0,3"]) == 0.3
    assert parse_conf(["0.5"]) == 0.5
    assert parse_conf(["2"]) is None and parse_conf(["abc"]) is None and parse_conf([]) is None


def test_parse_allowed():
    assert parse_allowed("") == set()
    assert parse_allowed("12, 34") == {12, 34}


def test_decode_image_applies_exif_rotation():
    im = Image.new("RGB", (40, 20), (255, 0, 0))
    exif = im.getexif()
    exif[0x0112] = 6  # rotate 90° on display
    buf = io.BytesIO()
    im.save(buf, "JPEG", exif=exif)
    bgr = decode_image(buf.getvalue())
    assert bgr.shape == (40, 20, 3)
    assert bgr[10, 10, 2] > 200 and bgr[10, 10, 0] < 50  # red stays red in BGR


def test_caption():
    assert caption(12, 0.4, 1.23).startswith("Товаров на фото: 12")
