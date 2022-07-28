from pathlib import Path

import cv2

from sr.decode import decode


def test_decode():
    data_dir = Path(__file__).parent / "data"

    image = cv2.imread(str(data_dir / "hello.png"))
    result = decode(image)
    assert result.message.rstrip() == "Hello world!"

    image = cv2.imread(str(data_dir / "hello1.jpg"))
    result = decode(image)
    assert result.message.rstrip() == "Hello world!"

    image = cv2.imread(str(data_dir / "hello2.jpg"))
    result = decode(image)
    assert result.message.rstrip() == "Hello world!"

    image = cv2.imread(str(data_dir / "hello3.jpg"))
    result = decode(image)
    assert result.message.rstrip() == "Hello world!"

    image = cv2.imread(str(data_dir / "empty.png"))
    result = decode(image)
    assert result.message.rstrip() == ""
