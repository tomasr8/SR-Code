from collections import namedtuple

import cv2
import numpy as np

from qr.lib import BLACK, WHITE

ORANGE = (0, 140, 240)
TEXT_COLOR = (155, 0, 255)

Contour = namedtuple("Contour", ["shape", "area"])


def find_contours(image, min_area=1e4, epsilon=0.05):
    contours, _ = cv2.findContours(image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = [simplify_contour(c, epsilon) for c in contours]
    contours = [c for c in contours if (
        cv2.contourArea(c) > min_area and  # Must not be too small
        cv2.isContourConvex(c)  # Must be convex since it's a square/rectangle
    )]
    contours = [Contour(c[:, 0, :], cv2.contourArea(c)) for c in contours]
    # The simplified contour must have 4 vertices
    contours = [c for c in contours if c.shape.shape[0] == 4]
    return contours


def simplify_contour(contour, epsilon):
    arc_length = cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon*arc_length, True)


def make_black_and_white(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Anything that is in this range is considered black
    return ~cv2.inRange(grayscale, 0, 150)


def warp_image(image, contour, dims):
    px = dims.pixels_per_square
    size = dims.squares_px
    inner_size = size-2*px

    pixel_coordinates = np.float64([
        [0, 0],
        [0, inner_size],
        [inner_size, inner_size],
        [inner_size, 0]
    ])

    H, _ = cv2.findHomography(contour, pixel_coordinates)
    image = cv2.warpPerspective(image, H, (inner_size, inner_size), flags=cv2.INTER_LINEAR)
    # https://pyimagesearch.com/2021/04/28/opencv-smoothing-and-blurring/
    image = cv2.medianBlur(image, 3)

    image_with_border = np.zeros((size, size), dtype=np.uint8)
    image_with_border[px:size-px, px:size-px] = image
    return image_with_border


def create_circular_mask(radius, size):
    Y, X = np.ogrid[:size, :size]
    c = size/2
    dist_from_center = np.sqrt((X - c)**2 + (Y - c)**2)
    mask = dist_from_center <= radius
    return mask


def has_color(selection, color, threshold):
    total_area = np.prod(selection.shape)
    color_area = np.sum(selection == color)
    return (color_area/total_area) > threshold


def is_black(selection, threshold=0.6):
    return has_color(selection, BLACK, threshold)


def is_white(selection, threshold=0.6):
    return has_color(selection, WHITE, threshold)


def resize_with_aspect_ratio(image, width):
    (h, w) = image.shape[:2]
    r = width / w
    dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


def draw_text(image, text, pos, color, font_size=1, thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        image,
        text,
        pos,
        font,
        font_size,
        color,
        thickness,
    )


def draw_contours(image, contours, color, thickness):
    for contour in contours:
        cv2.polylines(image, [contour], True, color, thickness)
