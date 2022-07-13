import numpy as np
import cv2
import click

from qr.util import encode_string
from qr.lib import QrCodeDimensions, WHITE, BLACK
from qr.image import QrCodeImage

class QrCodeGenerator:
    def __init__(self, dims: QrCodeDimensions):
        self.dims = dims
        cv_image = WHITE * np.ones((dims.squares_px, dims.squares_px), dtype=np.uint8)
        self.image = QrCodeImage(cv_image, dims)

    def _create_circular_mask(self, radius, size):
        Y, X = np.ogrid[:size, :size]
        c = size/2
        dist_from_center = np.sqrt((X - c)**2 + (Y - c)**2)
        mask = dist_from_center <= radius
        return mask

    # def draw_square(self, i, j):
    #     self[i:i+1, j:j+1] = BLACK

    def draw_start_corner(self):
        # |█| |
        # | | |
        self.image[2, 2] = BLACK
        # self.draw_square(2, 2)
        # self.draw_square(3, 2)
        # self.draw_square(2, 3)


    def draw_directional_corner(self):
        # |█| |
        # |█|█|
        self.image[-4, 2] = BLACK
        self.image[-3, 2] = BLACK
        self.image[-3, 3] = BLACK
        # self.draw_square(-4, 2)
        # self.draw_square(-3, 2)
        # self.draw_square(-3, 3)
        # self.draw_square(-5, 2)
        # self.draw_square(-3, 4)

    def draw_circles(self):
        big_circle_mask = self._create_circular_mask(
            self.dims.big_circle_radius_px, self.dims.squares_px)
        self.image._image[big_circle_mask] = BLACK

        small_circle_mask = self._create_circular_mask(
            self.dims.small_circle_radius_px, self.dims.squares_px)
        self.image._image[small_circle_mask] = WHITE

    def draw_outer_border(self):
        self.image[0, :] = BLACK
        self.image[-1:, :] = BLACK
        self.image[:, 0] = BLACK
        self.image[:, -1:] = BLACK


def generate_qr(message, size):
    qr = QrCodeGenerator(dims=QrCodeDimensions(pixels_per_square=size))
    assert len(message) * (6*3) <= qr.dims.max_data_size, f"{len(message)} characters won't fit in the QR code"
    data = encode_string(message)

    qr.draw_outer_border()
    qr.draw_start_corner()
    qr.draw_directional_corner()
    qr.draw_circles()

    for i, (y, x) in enumerate(qr.dims.data_squares()):
        if data[i] == 1:
            qr.image[y, x] = BLACK

    return cv2.cvtColor(qr.image._image, cv2.COLOR_GRAY2RGB)


@click.command()
@click.option("--size", "-s", type=int, default=25, help="Size of one square in pixels")
@click.option("--message", "-m", type=str, required=True, help="The message to encode")
@click.argument("output_file")
def generate(size, message, output_file):
    image = generate_qr(message, size)
    cv2.imwrite(output_file, image)    


if __name__ == '__main__':
    generate()
