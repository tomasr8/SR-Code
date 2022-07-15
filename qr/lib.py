from enum import Enum

WHITE = 255
BLACK = 0


class Corner(Enum):
    TOP_LEFT = 1
    BOTTOM_LEFT = 2
    TOP_RIGHT = 3
    BOTTOM_RIGHT = 4


class QrCodeDimensions:
    """Stores the dimensions of all the parts of the QR code

    Can convert between squares and pixels
    """

    def __init__(self, pixels_per_square=40):
        self.pixels_per_square = pixels_per_square
        self.squares = 24
        self.small_circle_radius = 1
        self.big_circle_radius = 2
        self.center = self.squares // 2
        self.reserved_outer_border = 3
        self.reserved_inner_radius = self.big_circle_radius + 1

        self._convertible_keys = {
            'squares', 'small_circle_radius', 'big_circle_radius', 'center'}

    def __getattr__(self, key):
        # Allows to dynamically convert squares to pixels by getting `{key}_px`
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

    @property
    def max_data_size(self):
        return len(list(self.data_squares()))

    @property
    def corners(self):
        return {
            Corner.TOP_LEFT: [[2, 2], [3, 2], [2, 3]],
            Corner.BOTTOM_LEFT: [[-3, 2], [-4, 2], [-3, 3]],
            Corner.TOP_RIGHT: [[2, -3], [2, -4], [3, -3]],
            Corner.BOTTOM_RIGHT: [[-3, -3], [-4, -3], [-3, -4]],
        }

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
