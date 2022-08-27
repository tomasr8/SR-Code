from pathlib import Path

import cv2

from sr.decode import decode


data_dir = Path(__file__).parent.parent / "examples"


def _has_message(result, message):
    return any(c.message.rstrip() == message for c in result.contours)


def _assert_message(name, msgs):
    image = cv2.imread(str(data_dir / name))
    result = decode(image)
    assert result.success
    if not isinstance(msgs, list):
        msgs = [msgs]
    for msg in msgs:
        assert _has_message(result, msg)


def test_decode():
    _assert_message("hello.png", "Hello world!")
    _assert_message("hello2.png", "Hello world!")
    _assert_message("empty.png", "")
    _assert_message("lena.png", "Hello world!")
    _assert_message("lena2.png", "Hello world!")
    _assert_message("codeception.png", ["Hello world!", "This works!", "Top corner!"])
    _assert_message("codeception2.png", ["Hello world!", "This works!", "Top corner!"])
    _assert_message("colors.png", "I can do colors!")
    _assert_message("colors2.png", "I can do colors!")
    _assert_message("all.png", ["I can do colors!", "This works!"])
