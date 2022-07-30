from pathlib import Path

import cv2

from sr.decode import decode


def _has_message(result, message):
    return any(c.message.rstrip() == message for c in result.contours)


def test_decode():
    data_dir = Path(__file__).parent / "data"

    image = cv2.imread(str(data_dir / "hello.png"))
    result = decode(image, find_all=True, use_bw=True)
    assert result.success
    assert _has_message(result, "Hello world!")

    image = cv2.imread(str(data_dir / "hello1.jpg"))
    result = decode(image, find_all=True, use_bw=True)
    assert result.success
    assert _has_message(result, "Hello world!")

    image = cv2.imread(str(data_dir / "hello2.jpg"))
    result = decode(image, find_all=True, use_bw=True)
    assert result.success
    assert _has_message(result, "Hello world!")

    image = cv2.imread(str(data_dir / "hello3.jpg"))
    result = decode(image, find_all=True, use_bw=True)
    assert result.success
    assert _has_message(result, "Hello world!")

    image = cv2.imread(str(data_dir / "empty.png"))
    result = decode(image, find_all=True, use_bw=True)
    assert result.success
    assert _has_message(result, "")
