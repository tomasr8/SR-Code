from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import List

import click
import cv2
import numpy as np

from qr.image import BLACK, WHITE, create_circular_mask
from qr.message import (CHARACTERS, DUPLICATION_FACTOR, LETTER_SIZE,
                        encode_message)


class Corner(Enum):
    TOP_LEFT = 1
    TOP_RIGHT = 3
    BOTTOM_RIGHT = 4
    BOTTOM_LEFT = 2


class QrCodeDimensions:
    """Stores the dimensions of all the parts of the QR code

    Can convert between squares and pixels
    """


@contextmanager
def video_capture(file=0):
    cap = cv2.VideoCapture(file)

    if not cap.isOpened():
        raise Exception(f"Failed to open file: {file}")

    try:
        yield cap
    finally:
        cap.release()


class DecodeError(Exception):
    pass


@dataclass
class DecodeResult:
    message: str = None
    bw_visualization: np.ndarray = None
    contour_visualization: np.ndarray = None
    decode_visualization: np.ndarray = None
    errors: List[str] = field(default_factory=list)

    def __repr__(self):
        result = click.style('Success', fg='green') if self.message else click.style('Failure', fg='red')
        return (
            f"Decode result: {result}\n" +
            f"- Message: {self.message}" if self.message else
            '\n'.join([f'- Contour {i+1}: {error}' for i, error in enumerate(self.errors)])
        )


class SRCode:
    def __init__(self, pixels_per_square, image=None):
        self.pixels_per_square = pixels_per_square
        self.squares = 24
        self.small_ring_radius = 1
        self.large_ring_radius = 2
        self.center = self.squares // 2
        self.reserved_outer_border = 3
        self.reserved_inner_radius = self.large_ring_radius + 1
        size_px = self.px(self.squares)
        self.image = image if image is not None else np.full((size_px, size_px), WHITE, dtype=np.uint8)

    def __getitem__(self, key):
        key = self._slice_to_px(key)
        return self.image[key]

    def __setitem__(self, key, value):
        key = self._slice_to_px(key)
        self.image[key] = value

    def _slice_to_px(self, key):
        if isinstance(key, int):
            return slice(self.px(key), self.px(key+1))
        elif isinstance(key, slice):
            return slice(
                self.px(key.start) if key.start is not None else None,
                self.px(key.stop) if key.stop is not None else None,
                self.px(key.step) if key.step is not None else None,
            )
        elif isinstance(key, tuple):
            return tuple(self._slice_to_px(item) for item in key)
        else:
            raise Exception(f"Invalid slice argument: {key}")

    def px(self, value, to_int=True):
        if isinstance(value, Iterable):
            return [self.px(item) for item in value]
        elif value < 0:
            value = (self.squares + value) * self.pixels_per_square
        else:
            value = value * self.pixels_per_square
        return round(value) if to_int else value

    def data_squares(self):
        for col in range(self.squares):
            for row in range(self.squares):
                if self._is_reserved(row, col):
                    continue
                yield row, col

    @property
    def max_data_size(self):
        return len(list(self.data_squares()))

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

    @property
    def corners(self):
        return {
            Corner.TOP_LEFT: [[3, 2], [2, 2], [2, 3]],
            Corner.TOP_RIGHT: [[2, -4], [2, -3], [3, -3]],
            Corner.BOTTOM_RIGHT: [[-4, -3], [-3, -3], [-3, -4]],
            Corner.BOTTOM_LEFT: [[-3, 3], [-3, 2], [-4, 2]],
        }


class GridImage:
    """A container for an opencv image

    Makes it possible to directly index individual squares.
    For example instead of doing:
        image[i*px:(i+1)*px, j*px:(j+1)*px]
    you can instead do:
        image[i, j]

    Slices are also supported:
        image[5:, 3:-1]
    """


class SRCodeGenerator(SRCode):
    def _draw_start_corner(self):
        # |█|█|
        # |█| |
        self[3, 2] = BLACK
        self[2, 2] = BLACK
        self[2, 3] = BLACK

    def _draw_circles(self):
        size = self.px(self.squares)
        large_ring_mask = create_circular_mask(
            self.px(self.large_ring_radius), size)
        self.image[large_ring_mask] = BLACK

        small_ring_mask = create_circular_mask(
            self.px(self.small_ring_radius), size)
        self.image[small_ring_mask] = WHITE

    def _draw_outer_border(self):
        self[0, :] = BLACK
        self[-1:, :] = BLACK
        self[:, 0] = BLACK
        self[:, -1:] = BLACK

    def generate(self, message):
        data = encode_message(message)

        self._draw_outer_border()
        self._draw_start_corner()
        self._draw_circles()

        for i, (y, x) in enumerate(self.data_squares()):
            if data[i] == 1:
                self[y, x] = BLACK

        return cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)

    @property
    def max_message_length(self):
        return self.max_data_size // (LETTER_SIZE * DUPLICATION_FACTOR)

    def validate_message(self, message):
        if len(message) > self.max_message_length:
            return f"Message too long ({len(message)}), maximum is {self.max_message_length}"

        if not all(letter in CHARACTERS for letter in message):
            return f"Invalid characters in the message\nAllowed characters: {CHARACTERS}"
