from pathlib import Path

import cv2
import pytest

from sr.code import EncodeError, SRCodeGenerator


def _encode_image(image):
    _, buffer = cv2.imencode(".png", image)
    return buffer.tobytes()


def test_generate(snapshot):
    snapshot.snapshot_dir = Path(__file__).parent / 'data'
    sr = SRCodeGenerator(size=600)

    image = sr.generate("Hello world!")
    snapshot.assert_match(_encode_image(image), 'hello.png')

    image = sr.generate("")
    snapshot.assert_match(_encode_image(image), 'empty.png')

    with pytest.raises(EncodeError):
        sr.generate("Invalid char: @")

    with pytest.raises(EncodeError):
        sr.generate("This message is too long")
