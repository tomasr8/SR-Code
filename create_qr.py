import numpy as np
import cv2

from util import encode_string

__all__ = ['create_qr']


WHITE = 255
BLACK = 0

class QrDimensions:
    def __init__(self, pixels_per_square=40):
        self.pixels_per_square = pixels_per_square
        self.squares = 24
        self.small_circle_radius = 1
        self.big_circle_radius = 2
        self.center = self.squares / 2
        self.reserved_outer_border = 3
        self.reserved_inner_radius = self.big_circle_radius + 1

        self._convertible_keys = {
            'squares', 'small_circle_radius', 'big_circle_radius', 'center'}

    def __getattr__(self, key):
        # allows to dynamically convert squares to pixels by getting `{key}_px`
        if key.endswith('_px') and key[:-3] in self._convertible_keys:
            key = key[:-3]
            return self.px(getattr(self, key))
        else:
            return getattr(self, key)

    def px(self, value):
        return value * self.pixels_per_square

    def data_squares(self):
        for col in range(self.squares):
            for row in range(self.squares):
                if self._is_reserved(row, col):
                    continue
                yield row, col

    def max_data_size(self):
        return len(list(self.data_squares())) // 3

    def _is_reserved(self, x, y):
        outer_border = (
            x < self.reserved_outer_border or
            y < self.reserved_outer_border or
            x >= (self.squares - self.reserved_outer_border) or
            y >= (self.squares - self.reserved_outer_border)
        )
        inner_circle = (
            x >= (self.center - self.reserved_inner_radius) and
            x < (self.center + self.reserved_inner_radius) and
            y >= (self.center - self.reserved_inner_radius) and
            y < (self.center + self.reserved_inner_radius)
        )
        return outer_border or inner_circle


class QrCode:
    def __init__(self, dimensions: QrDimensions):
        self.dimensions = dimensions
        # All white
        self.image = WHITE * np.ones((dimensions.squares_px, dimensions.squares_px), dtype=np.uint8)

    def _create_circular_mask(self, radius, size):
        Y, X = np.ogrid[:size, :size]
        c = size/2
        dist_from_center = np.sqrt((X - c)**2 + (Y - c)**2)
        mask = dist_from_center <= radius
        return mask

    def draw_square(self, i, j):
        self[i:i+1, j:j+1] = BLACK

    def draw_start_corner(self):
        # |█| |
        # | | |
        self.draw_square(2, 2)

    def draw_directional_corner(self):
        # |█| |
        # |█|█|
        self.draw_square(-4, 2)
        self.draw_square(-3, 2)
        self.draw_square(-3, 3)

    def draw_circles(self):
        big_circle_mask = self._create_circular_mask(
            self.dimensions.big_circle_radius_px, self.dimensions.squares_px)
        self.image[big_circle_mask] = BLACK

        small_circle_mask = self._create_circular_mask(
            self.dimensions.small_circle_radius_px, self.dimensions.squares_px)
        self.image[small_circle_mask] = WHITE

    def draw_inner_border(self):
        self[1:-1, 1:2] = BLACK
        self[1:-1, -2:-1] = BLACK
        self[1:2, 1:-1] = BLACK
        self[-2:-1, 1:-1] = BLACK

    def __getitem__(self, key):
        key = self._key_to_px(key)
        return self.image[key]

    def __setitem__(self, key, value):
        key = self._key_to_px(key)
        self.image[key] = value

    def _key_to_px(self, key):
        if isinstance(key, int):
            return self.dimensions.px(key)
        elif isinstance(key, slice):
            return slice(
                self.dimensions.px(key.start) if key.start is not None else None,
                self.dimensions.px(key.stop) if key.stop is not None else None,
                self.dimensions.px(key.step) if key.step is not None else None,
            )
        elif isinstance(key, tuple):
            return tuple(self._key_to_px(item) for item in key)
        else:
            raise Exception(f"Invalid slice argument: {key}")


def create_qr(string: str):
    data = encode_string(string)
    assert len(data) == 96 * 3

    qr = QrCode(dimensions=QrDimensions(pixels_per_square=20))
    qr.draw_inner_border()
    qr.draw_start_corner()
    qr.draw_directional_corner()
    qr.draw_circles()

    print(qr.dimensions.max_data_size())

    for i, (x, y) in enumerate(qr.dimensions.data_squares()):
        if data[i] == 0:
            qr.draw_square(x, y)

    return cv2.cvtColor(qr.image, cv2.COLOR_GRAY2RGB)


if __name__ == '__main__':
    qr = create_qr("Hello world!")

    cv2.imshow("frame", qr)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    # cv2.imwrite("qr5.jpg", qr)
