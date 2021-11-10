import matplotlib.pyplot as plt
import numpy as np
import cv2

__all__ = ['create_qr']


SQUARES = 24  # squares per side
PX = 40  # pixels per square
TOTAL_SIZE = SQUARES * PX
CENTER = TOTAL_SIZE//2

DATA_ROWS = SQUARES - 6
DATA_COLS = DATA_ROWS

BIG_CIRCLE_RADIUS = 2 * PX
SMALL_CIRCLE_RADIUS = 1 * PX


def create_circular_mask(x, y, r, w, h):
    Y, X = np.ogrid[:h, :w]

    dist_from_center = np.sqrt((X - x)**2 + (Y - y)**2)

    mask = dist_from_center <= r
    return mask


def draw_square(image, i, j):
    x = i*PX
    y = j*PX

    xp = (i+1)*PX
    yp = (j+1)*PX

    image[y:yp, x:xp] = 0


def draw_black_inner_border(image):
    image[PX:-PX, PX:2*PX] = 0
    image[PX:-PX, -2*PX:-PX] = 0
    image[PX:2*PX, PX:-PX] = 0
    image[-2*PX:-PX, PX:-PX] = 0


def draw_start_corner(image):
    draw_square(image, 2, 2)


def draw_end_corner(image):
    draw_square(image, 2, SQUARES - 4)
    draw_square(image, 2, SQUARES - 3)
    draw_square(image, 3, SQUARES - 3)


def draw_circles(image):
    big_circle_mask = create_circular_mask(CENTER, CENTER, BIG_CIRCLE_RADIUS, TOTAL_SIZE, TOTAL_SIZE)
    image[big_circle_mask] = 0

    small_circle_mask = create_circular_mask(CENTER, CENTER, SMALL_CIRCLE_RADIUS, TOTAL_SIZE, TOTAL_SIZE)
    image[small_circle_mask] = 255


def encode_string(string: str):
    assert len(string) <= 16

    data = list(string)

    if len(data) < 16:
        data += [" "] * (16 - len(data))

    BASE64 = " !0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

    encoded = []
    for letter in data:
        index = BASE64.find(letter)
        binary_string = "{0:06b}".format(index)

        for b in binary_string:
            encoded.append(int(b))

    return encoded + encoded + encoded


def create_qr(string: str):
    data = encode_string(string)
    assert len(data) == 96 * 3

    image = np.ones((SQUARES * PX, SQUARES * PX), dtype=np.uint8)
    image[:] = 255

    draw_black_inner_border(image)
    draw_start_corner(image)
    draw_end_corner(image)
    draw_circles(image)

    sx = 3
    sy = 3

    k = 0
    for i in range(DATA_ROWS):
        for j in range(DATA_COLS):
            x = (sx + i)
            y = (sy + j)

            if (x >= 9 and x <= 14) and (y >= 9 and y <= 14): # reserved space for the circles
                continue

            if data[k] == 0:
                draw_square(image, x, y)

            k += 1

    qr = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return qr


if __name__ == '__main__':
    qr = create_qr("Hello world!")

    cv2.imshow("frame", qr)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    # cv2.imwrite("qr5.jpg", qr)
