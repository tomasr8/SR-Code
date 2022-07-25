import cv2

from qr.util import DUPLICATION_FACTOR, LETTER_SIZE, encode_string, CHARACTERS
from qr.lib import SRCode, WHITE, BLACK
from qr.image import create_circular_mask


class SRCodeGenerator(SRCode):
    def _draw_start_corner(self):
        # |█| |
        # |█|█|
        self[3, 2] = BLACK
        self[2, 2] = BLACK
        self[2, 3] = BLACK

    def _draw_circles(self):
        size = self.px(self.squares)
        large_circle_mask = create_circular_mask(
            self.px(self.large_circle_radius), size)
        self.image[large_circle_mask] = BLACK

        small_circle_mask = create_circular_mask(
            self.px(self.small_circle_radius), size)
        self.image[small_circle_mask] = WHITE

    def _draw_outer_border(self):
        self[0, :] = BLACK
        self[-1:, :] = BLACK
        self[:, 0] = BLACK
        self[:, -1:] = BLACK

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

    @property
    def max_message_length(self):
        return self.max_data_size // (LETTER_SIZE * DUPLICATION_FACTOR)

    def validate_message(self, message):
        if len(message) > self.max_message_length:
            return f"Message too long ({len(message)}), maximum is {self.max_message_length}"

        if not all(letter in CHARACTERS for letter in message):
            return f"Invalid characters in the message\nAllowed characters: {CHARACTERS}"
