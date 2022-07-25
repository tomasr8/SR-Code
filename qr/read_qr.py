import numpy as np
import cv2

from qr.util import decode_data
from qr.lib import DecodeResult, SRCode, Corner, WHITE, BLACK
from qr.image import TEXT_COLOR, draw_contours, draw_text, find_contours, make_black_and_white, resize_with_aspect_ratio, warp_image, create_circular_mask, is_black, is_white


class SRCodeReader(SRCode):
    def _verify_inner_circles(self):
        big_circle_mask = create_circular_mask(
            self.px(self.large_circle_radius), self.px(self.squares))
        small_circle_mask = create_circular_mask(
            self.px(self.small_circle_radius), self.px(self.squares))

        small_circle = self.image[small_circle_mask]
        big_circle = self.image[big_circle_mask & (~small_circle_mask)]

        if not is_white(small_circle, threshold=0.6):
            return "Failed to find small circle"
        elif not is_black(big_circle, threshold=0.6):
            return "Failed to find big circle"

    def _find_start_corner(self):
        for name, positions in self.corners.items():
            corner = [self[y, x] for (y, x) in positions]
            if all(is_black(square) for square in corner):
                return name

    def normalize_orientation(self):
        start_corner = self._find_start_corner()
        if start_corner is None:
            return "Failed to find the starting corner"

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

    def visualize_outer_border(self, image, color, thickness):
        cv2.rectangle(image, self.px([1, 1]), self.px([-1, -1]), color,
                      thickness)

    def visualize_inner_circles(self, image, color, thickness):
        center = (self.px(self.center), self.px(self.center))
        cv2.circle(image, center, self.px(self.large_circle_radius), color,
                   thickness)
        cv2.circle(image, center, self.px(self.small_circle_radius), color,
                   thickness)

    def visualize_start_corner(self, image, color, thickness):
        points = np.array(
            self.px([(2, 2), (2, 4), (3, 4), (3, 3), (4, 3), (4, 2)]))
        cv2.polylines(image, [points], True, color, thickness)

    def visualize_data_points(self, image):
        for color, (y, x) in zip(self._read_colors(), self.data_squares()):
            pos = self.px([x + 0.5, y + 0.5], to_int=True)
            cv2.circle(image, pos, 3, [140, 240, 0], -1)
            text = str(0 if color == WHITE else 1)
            draw_text(image, text, self.px([x, y + 1]), TEXT_COLOR, 0.3, 1)

    def visualize_message(self, image, message, font_size=1.2, thickness=2):
        pos = self.px([4.5, -1.5], to_int=True)
        draw_text(image, message, pos, TEXT_COLOR, font_size, thickness)

    def visualize_error(self, image, error):
        self.visualize_message(image, error, font_size=0.6, thickness=2)

    def visualize(self, image):
        orange = (0, 140, 240)
        self.visualize_outer_border(image, orange, 2)
        self.visualize_inner_circles(image, orange, 2)
        self.visualize_corners(image, orange, 2)
        self.visualize_points(image)
        # self.visualize_message(image, message)

    def _read_colors(self):
        return [
            WHITE if is_white(self[y, x], threshold=0.5) else BLACK
            for (y, x) in self.data_squares()
        ]

    def read_data(self):
        raw_data = [
            0 if color == WHITE else 1 for color in self._read_colors()
        ]
        return decode_data(raw_data)


def read_qr(image):
    bw_image = make_black_and_white(image)
    contours = find_contours(bw_image)
    result = DecodeResult(contour_visualization=image.copy())

    draw_contours(result.contour_visualization, [c.shape for c in contours],
                  (0, 140, 240), 15)
    result.contour_visualization = resize_with_aspect_ratio(
        result.contour_visualization, 500)

    for contour in contours:
        pixels_per_square = np.sqrt(contour.area) // 24
        warped_bw_image = warp_image(bw_image, contour.shape, dims)
        result.qr_code_visualization = cv2.cvtColor(warped_bw_image.copy(),
                                                    cv2.COLOR_GRAY2RGB)
        qr = SRCodeReader(pixels_per_square, image=warped_bw_image)

        orange = (0, 140, 240)
        qr.visualize_outer_border(result.qr_code_visualization, orange, 2)
        qr.visualize_inner_circles(result.qr_code_visualization, orange, 2)

        if error := qr._check_inner_circles():
            result.errors.append(error)
            qr.visualize_error(result.qr_code_visualization, error)
            continue

        if error := qr.normalize_orientation():
            qr.visualize_error(result.qr_code_visualization, error)
            result.errors.append(error)
            continue

        result.message = qr.read_data()
        result.qr_code_visualization = cv2.cvtColor(qr.image._image.copy(),
                                                    cv2.COLOR_GRAY2RGB)
        qr.visualize_outer_border(result.qr_code_visualization, orange, 2)
        qr.visualize_inner_circles(result.qr_code_visualization, orange, 2)
        qr.visualize_corners(result.qr_code_visualization, orange, 2)
        qr.visualize_points(result.qr_code_visualization)
        qr.visualize_message(result.qr_code_visualization, result.message)
        break

    return result


if __name__ == '__main__':
    # read_qr("data/qr5_slanted.jpg")
    # read_qr("qr_new.jpg")
    # read_qr("annina.png")
    # read_qr("annina1.jpg")
    read_qr("annina2.jpg")
