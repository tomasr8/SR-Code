import numpy as np
import cv2

from qr.util import decode_data
from qr.lib import QrCodeDimensions, Corner, WHITE, BLACK
from qr.image import QrCodeImage, find_contours, make_black_and_white, resize_with_aspect_ratio, warp_image, create_circular_mask, is_black, is_white


class QrCodeReader:

    def __init__(self, dims: QrCodeDimensions, image, visualization_qr_code):
        self.dims = dims
        self.image = image
        self.visualization_qr_code = visualization_qr_code

    def _check_inner_circles(self):
        big_circle_mask = create_circular_mask(self.dims.big_circle_radius_px,
                                               self.dims.squares_px)
        small_circle_mask = create_circular_mask(
            self.dims.small_circle_radius_px, self.dims.squares_px)

        small_circle = self.image._image[small_circle_mask]
        big_circle = self.image._image[big_circle_mask & (~small_circle_mask)]
        return is_white(small_circle) and is_black(big_circle)

    def _find_corner_pattern(self, match_pattern):
        for name, positions in self.dims.corners.items():
            print(name)
            corner = [self.image[y, x] for (y, x) in positions]
            if match_pattern(*corner):
                return name

    def _find_start_corner(self):

        def is_start_corner(middle, left, right):
            # print(is_black(middle), is_white(left), is_white(right))
            return is_black(middle) and is_white(left) and is_white(right)

        return self._find_corner_pattern(is_start_corner)

    def _find_directional_corner(self):

        def is_directional_corner(middle, left, right):
            return is_black(middle) and is_black(left) and is_black(right)

        return self._find_corner_pattern(is_directional_corner)

    def normalize_orientation(self):
        start_corner = self._find_start_corner()
        if start_corner is None:
            return "Failed to find the starting corner"

        center = (self.dims.center_px, self.dims.center_px)
        if start_corner == Corner.BOTTOM_LEFT:
            M = cv2.getRotationMatrix2D(center, -90, 1.0)
        elif start_corner == Corner.TOP_RIGHT:
            M = cv2.getRotationMatrix2D(center, 90, 1.0)
        elif start_corner == Corner.BOTTOM_RIGHT:
            M = cv2.getRotationMatrix2D(center, 180, 1.0)

        if start_corner != Corner.TOP_LEFT:
            (h, w) = self.image._image.shape[:2]
            self.image._image = cv2.warpAffine(self.image._image, M, (w, h))

        directional_corner = self._find_directional_corner()
        if directional_corner is None:
            return "Failed to find the directional corner"
        elif directional_corner == Corner.BOTTOM_RIGHT:
            return "Directional corner found in the wrong position"
        if directional_corner == Corner.TOP_RIGHT:
            self.image._image = cv2.transpose(self.image._image)

    def _visualize_outer_border(self):
        px = self.dims.pixels_per_square
        size = self.dims.squares_px
        cv2.rectangle(self.visualization_qr_code, [px, px],
                      [size - px, size - px], [0, 140, 240], 2)

    def _visualize_inner_circles(self):
        center = (self.dims.center_px, self.dims.center_px)
        self.visualization_qr_code = cv2.circle(self.visualization_qr_code,
                                                center,
                                                self.dims.big_circle_radius_px,
                                                [0, 140, 240], 2)
        self.visualization_qr_code = cv2.circle(
            self.visualization_qr_code, center,
            self.dims.small_circle_radius_px, [0, 140, 240], 2)

    def _visualize_corners(self):
        px = self.dims.pixels_per_square
        size = self.dims.squares_px
        self.visualization_qr_code = cv2.rectangle(self.visualization_qr_code,
                                                   [2 * px, 2 * px],
                                                   [3 * px, 3 * px],
                                                   [0, 140, 240], 2)
        points = np.array([(2 * px, size - 4 * px), (2 * px, size - 2 * px),
                           (4 * px, size - 2 * px), (4 * px, size - 3 * px),
                           (3 * px, size - 3 * px), (3 * px, size - 4 * px)])
        self.visualization_qr_code = cv2.polylines(self.visualization_qr_code,
                                                   [points], True,
                                                   (0, 140, 240), 2)

    def _visualize_points(self):
        px = self.dims.pixels_per_square
        for color, (y, x) in zip(self._read_colors(),
                                 self.dims.data_squares()):
            cv2.circle(self.visualization_qr_code,
                       (int(x * px + px / 2), int(y * px + px / 2)), 3,
                       [140, 240, 0], -1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (x * px, y * px + px)
            fontScale = 0.3
            fontColor = (155, 0, 255)
            lineType = 1

            cv2.putText(
                self.visualization_qr_code,
                str(0 if color == WHITE else 1),
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType,
            )

    def _read_colors(self):
        return [
            WHITE if is_white(self.image[y, x], threshold=0.5) else BLACK
            for (y, x) in self.dims.data_squares()
        ]

    def read_data(self):
        raw_data = [
            0 if color == WHITE else 1 for color in self._read_colors()
        ]
        return decode_data(raw_data)


def _show(image, title="image"):
    cv2.imshow(title, image)
    while True:
        key = cv2.waitKey(1) & 0xFFf
        if key == ord("q"):
            break
    cv2.destroyWindow(title)


def read_qr(path):
    image = cv2.imread(path)
    bw_image = make_black_and_white(image)

    dims = QrCodeDimensions(pixels_per_square=30)
    contours = find_contours(bw_image)

    # contours, _ = cv2.findContours(bw_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # bw_image = cv2.cvtColor(bw_image, cv2.COLOR_GRAY2BGR)

    # cv2.drawContours(bw_image, contours, -1, (0, 255, 0), 2)
    # bw_image = cv2.resize(bw_image, (1280, 960), interpolation=cv2.INTER_AREA)

    # cv2.imshow("frame", bw_image)
    # while True:
    #     key = cv2.waitKey(1) & 0xFFf
    #     if key == ord("q"):
    #         break

    # return

    for contour in contours:
        warped_bw_image = warp_image(bw_image, contour, dims)
        visualization_original = image.copy()
        visualization_qr_code = cv2.cvtColor(warped_bw_image.copy(),
                                             cv2.COLOR_GRAY2RGB)

        visualization_original = cv2.polylines(visualization_original,
                                               [contour], True, (0, 140, 240),
                                               30)
        visualization_original = resize_with_aspect_ratio(
            visualization_original, 960)

        # _show(visualization_original, "Contours")
        # _show(warped_bw_image, "Warped")

        qr = QrCodeReader(dims=dims,
                          image=QrCodeImage(warped_bw_image, dims),
                          visualization_qr_code=visualization_qr_code)
        if not qr._check_inner_circles():
            print("INNER CIRCLE TEST FAILED")
            continue

        # _show(qr.visualization_qr_code, "Circles")
        if error := qr.normalize_orientation():
            print(error)
            continue

        qr.visualization_qr_code = cv2.cvtColor(qr.image._image.copy(),
                                                cv2.COLOR_GRAY2RGB)
        qr._visualize_outer_border()
        qr._visualize_inner_circles()
        qr._visualize_corners()
        qr._visualize_points()
        _show(qr.visualization_qr_code, "Final")
        # _show(qr.image._image, "Final")

        print(qr.read_data())


def read_qr_video(image):
    shape = image.shape
    orig_image = image

    black_lower = (0, 0, 0)
    black_upper = (180, 255, 70)

    # blurred = cv2.GaussianBlur(image, (11, 11), 0)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, black_lower, black_upper)
    # mask = cv2.erode(mask, None, iterations=2)
    # mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    image = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    for i, cnt in enumerate(contours):
        if (cv2.contourArea(cnt) < 100):
            continue

        epsilon = 0.05 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if not cv2.isContourConvex(approx):
            continue

        if approx.shape[0] == 4:
            rect = approx[:, 0, :]
            target = np.float64([[0, 0], [0, 880], [880, 880], [880, 0]])

            H = solve_homography(rect, target)

            image_h = cv2.warpPerspective(image, H, (shape[0], shape[1]))
            image_h = image_h[:881, :881]

            mask_h = cv2.warpPerspective(mask, H, (shape[0], shape[1]))
            mask_h = mask_h[:881, :881]

            if not is_main_square(mask_h, image_h, target):
                continue

            cv2.drawContours(orig_image, [cnt], -1, (0, 255, 0), 2)

            size = 880 / 22

            # warped = cv2.warpPerspective(orig_image, H, (shape[0], shape[1]))
            for i in range(22):
                for j in range(22):
                    cv2.circle(image_h, (int(size * i), int(size * j)), 3,
                               (100, 20, 200), -1)
            # unwarped = cv2.warpPerspective(warped, np.linalg.pinv(H), (shape[0], shape[1]))

            data = read_data(mask_h, image_h, size)
            image_h = cv2.resize(image_h, (440, 440),
                                 interpolation=cv2.INTER_AREA)
            orig_image[-440:, -440:] = image_h

            print(orig_image.shape[0] - 460, orig_image.shape[1] - 440)

            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (orig_image.shape[1] - 440,
                                      orig_image.shape[0] - 460)
            fontScale = 2
            fontColor = (155, 0, 255)
            lineType = 3

            cv2.putText(orig_image, data, bottomLeftCornerOfText, font,
                        fontScale, fontColor, lineType)

            return orig_image

    return orig_image


if __name__ == '__main__':
    # read_qr("data/qr5_slanted.jpg")
    # read_qr("qr_new.jpg")
    # read_qr("annina.png")
    # read_qr("annina1.jpg")
    read_qr("annina2.jpg")

