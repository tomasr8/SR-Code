import math
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import List

import click
import cv2
import numpy as np

from sr.image import (BLACK, GREEN, ORANGE, PURPLE, WHITE, create_circular_mask, get_optimal_font_scale,
                      get_optimal_thickness, is_black, is_white, make_black_and_white, vertical_concat, warp_image)
from sr.message import CHARACTERS, DUPLICATION_FACTOR, LETTER_SIZE, decode_data, encode_message


class Corner(Enum):
    TOP_LEFT = 1
    TOP_RIGHT = 2
    BOTTOM_RIGHT = 3
    BOTTOM_LEFT = 4


@contextmanager
def video_capture(file=0):
    cap = cv2.VideoCapture(file)
    if not cap.isOpened():
        raise Exception(f"Failed to open file: {file}")

    try:
        yield cap
    finally:
        cap.release()


class EncodeError(Exception):
    pass


class DecodeError(Exception):
    pass


@dataclass
class ContourResult:
    message: str = ''
    error: str = ''
    visualization: np.ndarray = None

    @property
    def success(self):
        return bool(self.message)

    def __str__(self):
        if self.success:
            return f"Message: {click.style(self.message, fg='green')}"
        else:
            return f"Error: {click.style(self.error, fg='red')}"


@dataclass
class DecodeResult:
    """Stores the decode result.

    It contains a result for each found contour.
    Each contour either has an error, or a message and a visualization.
    """
    time_ms: float = 0
    contour_visualization: np.ndarray = None
    contours: List[ContourResult] = field(default_factory=list)

    def __str__(self):
        result = click.style('Success', fg='green') if self.success else click.style('Failure', fg='red')
        time = click.style(f"{self.time_ms:.1f}ms", fg='yellow')
        contour_count = click.style(len(self.contours), fg='yellow')
        contours = [str(contour) for contour in self.contours]
        contours = '\n'.join([f'    {i+1}: {contour}' for i, contour in enumerate(contours)])

        return (
            f"Decode result: {result}\n" +
            f"- Total time: {time}\n" +
            f"- Contours found: {contour_count}\n" +
            contours
        )

    @property
    def success(self):
        return any(contour.success for contour in self.contours)

    def get_visualizations(self):
        return [contour.visualization for contour in self.contours if contour.visualization is not None]


class SRCode:
    """Base SR code class.

    Stores the dimensions of all the parts of the SR code and has
    some helper methods such to provide access to the data squares.

    This class also makes it possible to directly index individual squares.
    For example, instead of doing:
        image[i*px:(i+1)*px, j*px:(j+1)*px]
    you can do:
        image[i, j]

    Slices are also supported:
        image[5:, 3:-1]
    """
    def __init__(self, size, image=None):
        self.squares = 24
        self.pixels_per_square = max(1, round(size / self.squares))
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
        inner_rings = (
            x >= (self.center - self.reserved_inner_radius) and
            x < (self.center + self.reserved_inner_radius) and
            y >= (self.center - self.reserved_inner_radius) and
            y < (self.center + self.reserved_inner_radius)
        )
        return outer_border or inner_rings

    @property
    def corners(self):
        return {
            Corner.TOP_LEFT: [[3, 2], [2, 2], [2, 3]],
            Corner.TOP_RIGHT: [[2, -4], [2, -3], [3, -3]],
            Corner.BOTTOM_RIGHT: [[-4, -3], [-3, -3], [-3, -4]],
            Corner.BOTTOM_LEFT: [[-3, 3], [-3, 2], [-4, 2]],
        }


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
        self._validate_message(message)
        data = encode_message(message)

        self.image[:, :] = WHITE
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

    def _validate_message(self, message):
        if len(message) > self.max_message_length:
            raise EncodeError(f"Message too long ({len(message)}), maximum is {self.max_message_length}")

        if not all(letter in CHARACTERS for letter in message):
            raise EncodeError(f"Invalid characters in the message, allowed characters: {CHARACTERS}")


class SRCodeReader(SRCode):

    def _isolate_sr_code(self, contour):
        """Undistort the image & remove everything outside of the contour."""
        px = self.pixels_per_square
        size = self.px(self.squares)
        inner_size = self.px(self.squares - 2)
        pixel_coordinates = np.float64([[0, 0], [0, inner_size],
                                        [inner_size, inner_size],
                                        [inner_size, 0]])

        self.image = warp_image(self.image, contour.points, pixel_coordinates, inner_size)
        image_with_border = np.zeros((size, size, 3), dtype=np.uint8)
        image_with_border[px:(size - px), px:(size - px), :] = self.image
        # Convert to black and white only now when the rest of the image has been removed
        # This should improve the threshold estimation
        self.image = make_black_and_white(image_with_border)

    def _verify_inner_rings(self):
        """Make sure the image contains the inner rings."""
        large_ring_mask = create_circular_mask(
            self.px(self.large_ring_radius), self.px(self.squares))
        small_ring_mask = create_circular_mask(
            self.px(self.small_ring_radius), self.px(self.squares))

        small_ring = self.image[small_ring_mask]
        large_ring = self.image[large_ring_mask & (~small_ring_mask)]

        if not is_white(small_ring, threshold=0.6):
            raise DecodeError("Failed to find small ring")
        elif not is_black(large_ring, threshold=0.6):
            raise DecodeError("Failed to find large ring")

    def _find_start_corner(self):
        for name, positions in self.corners.items():
            corner = [self[y, x] for (y, x) in positions]
            if all(is_black(square) for square in corner):
                return name

    def _normalize_orientation(self):
        """Rotate the image so that the start corner is in the top left."""
        start_corner = self._find_start_corner()
        if start_corner is None:
            raise DecodeError("Failed to find the start corner")

        center = (self.px(self.center), self.px(self.center))
        if start_corner == Corner.BOTTOM_LEFT:
            M = cv2.getRotationMatrix2D(center, -90, 1.0)
        elif start_corner == Corner.TOP_RIGHT:
            M = cv2.getRotationMatrix2D(center, 90, 1.0)
        elif start_corner == Corner.BOTTOM_RIGHT:
            M = cv2.getRotationMatrix2D(center, 180, 1.0)

        if start_corner != Corner.TOP_LEFT:
            (h, w) = self.image.shape[:2]
            self.image = cv2.warpAffine(self.image, M, (w, h))

    def _visualize_outer_border(self, image, color=ORANGE, thickness=6):
        cv2.rectangle(image, self.px([1, 1]), self.px([-1, -1]), color,
                      thickness)

    def _visualize_inner_rings(self, image, color=ORANGE, thickness=6):
        center = (self.px(self.center), self.px(self.center))
        cv2.circle(image, center, self.px(self.large_ring_radius), color,
                   thickness)
        cv2.circle(image, center, self.px(self.small_ring_radius), color,
                   thickness)

    def _visualize_start_corner(self, image, color=ORANGE, thickness=6):
        points = np.array(
            self.px([(2, 2), (2, 4), (3, 4), (3, 3), (4, 3), (4, 2)]))
        cv2.polylines(image, [points], True, color, thickness)

    def _visualize_data_points(self, image):
        for color, (y, x) in zip(self._read_colors(), self.data_squares()):
            cv2.rectangle(image, self.px((x+0.4, y+0.4)), self.px((x+0.6, y+0.6)),
                          PURPLE if color == WHITE else GREEN, -1)

    def _visualize_message(self, image, message):
        """Append a message to the bottom of the visualization."""
        h, w = image.shape[:2]
        text_height, text_width = round(0.1*h), round(0.9*w)
        origin = (round(0.05*text_width), round(0.75*text_height))
        font = cv2.FONT_HERSHEY_DUPLEX
        thickness = get_optimal_thickness(text_width)
        scale = get_optimal_font_scale(message, text_width, thickness=thickness)
        text = np.zeros((text_height, w, 3), dtype=np.uint8)
        text = cv2.putText(text, message, origin, font, scale, ORANGE, thickness, cv2.LINE_AA)
        return vertical_concat(image, text)

    def _read_colors(self):
        return [
            WHITE if is_white(self[y, x], threshold=0.5) else BLACK
            for (y, x) in self.data_squares()
        ]

    def _read_data(self):
        raw_data = [
            0 if color == WHITE else 1 for color in self._read_colors()
        ]
        return decode_data(raw_data)

    def visualize_decoded(self, image, message):
        # Adjust the line thickness for the image resolution
        line_thickness = math.ceil(self.pixels_per_square / 5)
        self._visualize_outer_border(image, thickness=line_thickness)
        self._visualize_inner_rings(image, thickness=line_thickness)
        self._visualize_start_corner(image, thickness=line_thickness)
        self._visualize_data_points(image)
        return self._visualize_message(image, message)

    def decode(self, contour):
        """Decode data from region given by a contour.

        Returns the message as a simple string or
        raises `DecodeError` if decoding fails.
        """
        self._isolate_sr_code(contour)
        self._verify_inner_rings()
        self._normalize_orientation()
        return self._read_data()


def generate(size, message):
    """Generate and SR code from a message.

    Raises `EncodeError` if the message is too long or
    uses invalid characters.
    """
    sr = SRCodeGenerator(size=size)
    return sr.generate(message)
