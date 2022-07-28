from collections import namedtuple

import cv2
import numpy as np

WHITE = 255
BLACK = 0
ORANGE = (0, 140, 240)
GREEN = (0, 255, 0)
PURPLE = (155, 0, 255)

Contour = namedtuple("Contour", ["points", "area"])


def find_contours(image, min_area=1e5, epsilon=0.05):
    contours, _ = cv2.findContours(image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = [simplify_contour(c, epsilon) for c in contours]
    contours = [c for c in contours if (
        cv2.contourArea(c) > min_area and  # Must not be too small
        cv2.isContourConvex(c)  # Must be convex since it's a square/rectangle
    )]
    contours = [Contour(c[:, 0, :], cv2.contourArea(c)) for c in contours]
    # The simplified contour must have 4 vertices
    contours = [c for c in contours if c.points.shape[0] == 4]
    return contours


def simplify_contour(contour, epsilon):
    arc_length = cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon*arc_length, True)


def make_black_and_white(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Removes noise, which improves thresholding
    # https://pyimagesearch.com/2021/04/28/opencv-smoothing-and-blurring/
    grayscale = cv2.medianBlur(grayscale, 3)
    # https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
    _, threshold = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return threshold


def warp_image(image, source, target, size):
    H, _ = cv2.findHomography(source, target)
    return cv2.warpPerspective(image, H, (size, size), flags=cv2.INTER_LINEAR)


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


def draw_text(image, text, pos, color=PURPLE, font_size=1.0, thickness=2):
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


def draw_contours(image, contours, color=ORANGE, thickness=15):
    for contour in contours:
        cv2.polylines(image, [contour], True, color, thickness)


def horizontal_concat(left, right):
    h = max(left.shape[0], right.shape[0])
    w = left.shape[1] + right.shape[1]
    image = np.zeros((h, w, 3), dtype=np.uint8)
    image[:left.shape[0], :left.shape[1]] = left
    image[:right.shape[0], left.shape[1]:] = right
    return image


def pressed_quit(timeout_ms=0):
    return cv2.waitKey(timeout_ms) & 0xFF == ord('q')