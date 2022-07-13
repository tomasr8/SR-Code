import numpy as np
import cv2

from qr.util import decode_data
from qr.lib import WHITE, BLACK, QrCodeDimensions
from qr.image import QrCodeImage, find_contours, make_black_and_white, warp_image

class QrCodeReader:
    def __init__(self, dimensions: QrCodeDimensions):
        self.dimensions = dimensions
        self.image = None
        self.visualization = WHITE * np.ones((dimensions.squares_px, dimensions.squares_px), dtype=np.uint8)

    def _create_circular_mask(self, radius, size):
        Y, X = np.ogrid[:size, :size]
        c = size/2
        dist_from_center = np.sqrt((X - c)**2 + (Y - c)**2)
        mask = dist_from_center <= radius
        return mask

    def _has_color(self, selection, color, threshold):
        total_area = np.prod(selection.shape)
        color_area = np.sum(selection == color)
        return (color_area/total_area) > threshold

    def _is_black(self, selection, threshold=0.75):
        return self._has_color(selection, BLACK, threshold)

    def _is_white(self, selection, threshold=0.75):
        return self._has_color(selection, WHITE, threshold)

    def _check_inner_circles(self, warped_image):
        big_circle_mask = self._create_circular_mask(
            self.dimensions.big_circle_radius_px, self.dimensions.squares_px)
        small_circle_mask = self._create_circular_mask(
            self.dimensions.small_circle_radius_px, self.dimensions.squares_px)

        small_circle = warped_image[small_circle_mask]
        big_circle = warped_image[big_circle_mask & (~small_circle_mask)]

        # cv2.circle(self.visualization, (int(center[0]), int(center[1])), int(
        #     side_length * BIG_CIRCLE_RADIUS), [0, 255, 0])
        # cv2.circle(draw, (int(center[0]), int(center[1])), int(
        #     side_length * SMALL_CIRCLE_RADIUS), [0, 255, 0])

        return self._is_white(small_circle) and self._is_black(big_circle)

    def _find_corner_pattern(self, warped_image, pattern):
        corners = {
            'top_left': [[2, 2], [3, 2], [2, 3]],
            'bottom_left': [[-4, 2], [-3, 2], [-3, 3]],
            'top_right': [[1,-4], [1,-5], [2, -4]],
            'bottom_right': [[-4,-4], [-4, -5], [-5, -4]],
        }

        for name, positions in corners.items():
            middle, left, right = [self.get_square(*p) for p in positions]
            if pattern(middle, left, right):
                return name

    def _find_start_corner(self, warped_image):
        def is_start_corner(middle, left, right):
            return self._is_black(middle) and self._is_white(left) and self._is_white(right)
        return self._find_corner_pattern(warped_image, is_start_corner)
        
    def _find_directional_corner(self, warped_image):
        def is_directional_corner(middle, left, right):
            return self._is_black(middle) and self._is_black(left) and self._is_black(right)
        return self._find_corner_pattern(warped_image, is_directional_corner)        
    
    def orient(self, warped_image):
        start_corner = self._find_start_corner(warped_image)
        assert start_corner is not None

        if start_corner != 'top_left':
            if start_corner == 'bottom_left':
                M = cv2.getRotationMatrix2D((self.dimensions.center_px, self.dimensions.center_px), 90, 1.0)
            elif start_corner == 'top_right':
                M = cv2.getRotationMatrix2D((self.dimensions.center_px, self.dimensions.center_px), -90, 1.0)
            else:
                M = cv2.getRotationMatrix2D((self.dimensions.center_px, self.dimensions.center_px), 180, 1.0)

            (h, w) = warped_image.shape[:2]
            warped_image = cv2.warpAffine(warped_image, M, (w, h))
        
        directional_corner = self._find_directional_corner(warped_image)
        assert directional_corner != 'bottom_right'
        assert directional_corner is not None
        if directional_corner != 'bottom_left':
            warped_image = cv2.transpose(warped_image)
        return warp_image

    def read_data(self, warped_image):
        warped_image = self.orient(warped_image)

        raw_data = []
        for i, (x, y) in enumerate(self.dimensions.data_squares()):
            square = self.get_square(x, y)
            if self._is_white(square, threshold=0.5):
                raw_data.append(0)
                # cv2.circle(draw, (int(x*size + size/2), int(y*size + size/2)), 6, [100, 100, 0], -1)
            else:
                raw_data.append(1)
                # cv2.circle(draw, (int(x*size + size/2), int(y*size + size/2)), 6, [100, 0, 100], -1)
        return decode_data(raw_data)

    def get_square(self, i, j):
        return self.image[i, j]


def read_qr(path):
    image = cv2.imread(path)
    bw_image = make_black_and_white(image)

    qr = QrCodeReader(dimensions=QrCodeDimensions(pixels_per_square=25))
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
        warped_bw_image = warp_image(bw_image, contour, qr.dimensions)

        cv2.imshow("frame", warped_bw_image)
        while True:
            key = cv2.waitKey(1) & 0xFFf
            if key == ord("q"):
                break

        qr.image = QrCodeImage(warped_bw_image, qr.dimensions)

        if not qr._check_inner_circles(warped_bw_image):
            print("INNER CIRCLE TEST FAILED")
        warped_bw_image = qr.orient(warped_bw_image)

        print(qr.read_data(warped_bw_image))



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

    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    for i, cnt in enumerate(contours):
        if (cv2.contourArea(cnt) < 100):
            continue

        perimeter = cv2.arcLength(cnt, True)
        epsilon = 0.05*cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if not cv2.isContourConvex(approx):
            continue

        if approx.shape[0] == 4:
            rect = approx[:, 0, :]
            target = np.float64([
                [0, 0],
                [0, 880],
                [880, 880],
                [880, 0]
            ])

            H = solve_homography(rect, target)

            image_h = cv2.warpPerspective(image, H, (shape[0], shape[1]))
            image_h = image_h[:881, :881]

            mask_h = cv2.warpPerspective(mask, H, (shape[0], shape[1]))
            mask_h = mask_h[:881, :881]

            if not is_main_square(mask_h, image_h, target):
                continue

            cv2.drawContours(orig_image, [cnt], -1, (0, 255, 0), 2)

            size = 880/22

            # warped = cv2.warpPerspective(orig_image, H, (shape[0], shape[1]))
            for i in range(22):
                for j in range(22):
                    cv2.circle(image_h, (int(size*i), int(size*j)),
                               3, (100, 20, 200), -1)
            # unwarped = cv2.warpPerspective(warped, np.linalg.pinv(H), (shape[0], shape[1]))

            data = read_data(mask_h, image_h, size)
            image_h = cv2.resize(image_h, (440, 440), interpolation=cv2.INTER_AREA)
            orig_image[-440:, -440:] = image_h

            print(orig_image.shape[0] - 460, orig_image.shape[1] - 440)

            font                   = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (orig_image.shape[1] - 440, orig_image.shape[0] - 460)
            fontScale              = 2
            fontColor              = (155,0,255)
            lineType               = 3

            cv2.putText(orig_image, data, 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                lineType
            )

            return orig_image

    return orig_image


if __name__ == '__main__':
    # read_qr("data/qr5_slanted.jpg")
    read_qr("qr_new.jpg")

