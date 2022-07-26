import cv2
import numpy as np

from qr.code import (BLACK, WHITE, Corner, DecodeError, DecodeResult, SRCode,
                     video_capture)
from qr.image import (GREEN, ORANGE, PURPLE, create_circular_mask,
                      draw_contours, draw_text, find_contours, is_black,
                      is_white, make_black_and_white, resize_with_aspect_ratio,
                      warp_image)
from qr.message import decode_data


class SRCodeReader(SRCode):

    def _isolate_sr_code(self, contour):
        px = self.pixels_per_square
        size = self.px(self.squares)
        inner_size = self.px(self.squares - 2)
        pixel_coordinates = np.float64([[0, 0], [0, inner_size],
                                        [inner_size, inner_size],
                                        [inner_size, 0]])

        self.image = warp_image(self.image, contour, pixel_coordinates, inner_size)
        image_with_border = np.zeros((size, size), dtype=np.uint8)
        image_with_border[px:(size - px), px:(size - px)] = self.image
        self.image = image_with_border

    def _verify_inner_rings(self):
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

    def _visualize_outer_border(self, image, color=ORANGE, thickness=4):
        cv2.rectangle(image, self.px([1, 1]), self.px([-1, -1]), color,
                      thickness)

    def _visualize_inner_rings(self, image, color=ORANGE, thickness=4):
        center = (self.px(self.center), self.px(self.center))
        cv2.circle(image, center, self.px(self.large_ring_radius), color,
                   thickness)
        cv2.circle(image, center, self.px(self.small_ring_radius), color,
                   thickness)

    def _visualize_start_corner(self, image, color=ORANGE, thickness=4):
        points = np.array(
            self.px([(2, 2), (2, 4), (3, 4), (3, 3), (4, 3), (4, 2)]))
        cv2.polylines(image, [points], True, color, thickness)

    def _visualize_data_points(self, image):
        for color, (y, x) in zip(self._read_colors(), self.data_squares()):
            pos = self.px([x + 0.5, y + 0.5])
            cv2.circle(image, pos, 10, GREEN, -1)
            text = str(0 if color == WHITE else 1)
            draw_text(image, text, self.px([x, y + 1]), PURPLE, 0.9, 2)

    def visualize_decoded(self, image):
        self._visualize_outer_border(image)
        self._visualize_inner_rings(image)
        self._visualize_start_corner(image)
        self._visualize_data_points(image)

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

    def decode(self, contour):
        self._isolate_sr_code(contour.shape)
        self._verify_inner_rings()
        self._normalize_orientation()
        return self._read_data()


def decode(image):
    denoised = cv2.medianBlur(image, 3)
    bw = make_black_and_white(denoised)

    contours = find_contours(bw)
    result = DecodeResult(bw_visualization=resize_with_aspect_ratio(bw, 700),
                          contour_visualization=denoised.copy())
    draw_contours(result.contour_visualization, [c.shape for c in contours], ORANGE, 15)
    result.contour_visualization = resize_with_aspect_ratio(result.contour_visualization, 700)

    for contour in contours:
        pixels_per_square = round(np.sqrt(contour.area) / 24)
        sr = SRCodeReader(pixels_per_square, image=bw)

        try:
            result.message = sr.decode(contour)
        except DecodeError as e:
            result.errors.append(str(e))

        if result.message:
            result.decode_visualization = cv2.cvtColor(sr.image, cv2.COLOR_GRAY2RGB)
            sr.visualize_decoded(result.decode_visualization)
            result.decode_visualization = resize_with_aspect_ratio(result.decode_visualization, 700)
            break

    return result


def decode_video(file, return_on_read=False):
    with video_capture(file) as cap:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:

                # start = time.time()
                result = decode(frame)

                result.bw_visualization = cv2.cvtColor(result.bw_visualization, cv2.COLOR_GRAY2RGB)
                if result.decode_visualization is not None:
                    vis = cv2.resize(result.decode_visualization, (300, 300))
                    result.bw_visualization[:300, :300] = vis
                    result.bw_visualization[:300, :300] = vis

                print(">>", result.message)
                # if result.dec
                cv2.imshow('Frame', result.bw_visualization)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

            else:
                break
