import cv2
import numpy as np


class QrCodeImage:
    def __init__(self, image, dimensions):
        self._image = image
        self.dimensions = dimensions

    def __getitem__(self, key):
        key = self._key_to_px(key)
        return self._image[key]

    def __setitem__(self, key, value):
        key = self._key_to_px(key)
        self._image[key] = value

    def _key_to_px(self, key):
        if isinstance(key, int):
            if key < 0:
                key = self.dimensions.squares + key
            return slice(self.dimensions.px(key), self.dimensions.px(key+1))
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


def find_contours(image, min_area=1e4, epsilon=0.05):
    contours, _ = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = [simplify_contour(c, epsilon) for c in contours]
    contours = [c for c in contours if (
        cv2.contourArea(c) > min_area and # Must not be too small
        cv2.isContourConvex(c) # Must be convex since it's a square/rectangle
    )]
    contours = [c[:, 0, :] for c in contours]
    # The simplified contour must have 4 vertices
    contours = [c for c in contours if c.shape[0] == 4]
    return contours


def simplify_contour(contour, epsilon):
    arc_length = cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon*arc_length, True)


def make_black_and_white(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Anything that is in this range is considered black
    return ~cv2.inRange(grayscale, 0, 120)


def warp_image(image, contour, dimensions):
    px = dimensions.pixels_per_square
    size = dimensions.squares_px
    inner_size = size - 2*px

    pixel_coordinates = np.float64([
        [0, 0],
        [0, inner_size],
        [inner_size, inner_size],
        [inner_size, 0]
    ])

    # H = cv2.getPerspectiveTransform(contour, pixel_coordinates)
    H, _ = cv2.findHomography(contour, pixel_coordinates, method=0)
    image = cv2.warpPerspective(image, H, (inner_size, inner_size), flags=cv2.INTER_LINEAR)
    # https://pyimagesearch.com/2021/04/28/opencv-smoothing-and-blurring/
    image = cv2.medianBlur(image, 3)

    image_with_border = np.zeros((size, size), dtype=np.uint8)
    image_with_border[px:size-px, px:size-px] = image
    return image_with_border
