import numpy as np
import cv2

from qr.util import DUPLICATION_FACTOR, LETTER_SIZE, encode_string, CHARACTERS
from qr.lib import QrCodeDimensions, WHITE, BLACK
from qr.image import QrCodeImage, create_circular_mask


class QrCodeGenerator:
    def __init__(self, pixels_per_square):
        self.dims = QrCodeDimensions(pixels_per_square=pixels_per_square)
        self.reset()

    def _draw_start_corner(self):
        # |█| |
        # | | |
        self.image[2, 2] = BLACK

    def _draw_directional_corner(self):
        # |█| |
        # |█|█|
        self.image[-4, 2] = BLACK
        self.image[-3, 2] = BLACK
        self.image[-3, 3] = BLACK

    def _draw_circles(self):
        big_circle_mask = create_circular_mask(
            self.dims.big_circle_radius_px, self.dims.squares_px)
        self.image._image[big_circle_mask] = BLACK

        small_circle_mask = create_circular_mask(
            self.dims.small_circle_radius_px, self.dims.squares_px)
        self.image._image[small_circle_mask] = WHITE

    def _draw_outer_border(self):
        self.image[0, :] = BLACK
        self.image[-1:, :] = BLACK
        self.image[:, 0] = BLACK
        self.image[:, -1:] = BLACK

    def generate(self, message):
        data = encode_string(message)

        self._draw_outer_border()
        self._draw_start_corner()
        self._draw_directional_corner()
        self._draw_circles()

        for i, (y, x) in enumerate(self.dims.data_squares()):
            if data[i] == 1:
                self.image[y, x] = BLACK

        return cv2.cvtColor(self.image._image, cv2.COLOR_GRAY2RGB)

    def reset(self):
        cv_image = WHITE * np.ones((self.dims.squares_px, self.dims.squares_px), dtype=np.uint8)
        self.image = QrCodeImage(cv_image, self.dims)

    @property
    def max_message_length(self):
        return self.dims.max_data_size // (LETTER_SIZE * DUPLICATION_FACTOR)

    def validate_message(self, message):
        if len(message) > self.max_message_length:
            return f"Message too long ({len(message)}), maximum is {self.max_message_length}"

        if not all(letter in CHARACTERS for letter in message):
            return f"Invalid characters in the message\nCharacters supported: {CHARACTERS}"
