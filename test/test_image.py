import cv2
import numpy as np

from qr.image import QrCodeImage
from qr.lib import QrCodeDimensions

def test_indexing():
    dims = QrCodeDimensions(pixels_per_square=2)
    size = dims.squares_px
    image = QrCodeImage(np.zeros((size, size), np.uint8), dims)

    image[0, 0] = 1
    assert np.all(image._image[:2, :2] == 1)
    image[0, 0] = 0
    assert np.all(image._image[:2, :2] == 0)

    image[2:3, 0] = 1
    assert np.all(image._image[4:6, :2] == 1)
    image[2:3, 0] = 0
    assert np.all(image._image[4:6, :2] == 0)

    image[-1, 1:-1] = 1
    assert np.all(image._image[-2:, 2:-2] == 1)
    image[-1, 1:-1] = 0
    assert np.all(image._image[-2:, 2:-2] == 0)
