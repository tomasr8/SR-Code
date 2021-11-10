import matplotlib.pyplot as plt
import numpy as np
import cv2


SIZE_RATIO = 5

def is_black_square(square, threshold = 0.8):
    N = square.shape[0]
    total_area = N*N

    black_area = np.sum(square == 0)

    print("ratio", black_area/total_area)

    if (black_area/total_area) > threshold:
        return True
    else:
        return False

def is_white_square(square, threshold = 0.8):
    N = square.shape[0]
    total_area = N*N


    white_area = np.sum(square == 255)

    print("ratio", white_area/total_area)

    if (white_area/total_area) > threshold:
        return True
    else:
        return False


def is_main_square(image, draw, box):
    config = [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]
    ]

    a, b, c, d = box

    u = (b - a) / SIZE_RATIO
    v = (d - a) / SIZE_RATIO

    sx, sy = a
    dx = u[0] + v[0]
    dy = u[1] + v[1]

    print(dx, dy)

    for i in range(SIZE_RATIO):
        for j in range(SIZE_RATIO):

            x = int(sx + i*dx)
            y = int(sy + j*dy)

            xp = int(sx + (i+1)*dx)
            yp = int(sy + (j+1)*dy)

            if xp < x:
                [x, xp] = [xp, x]

            if yp < y:
                [y, yp] = [yp, y]

            print("center", x, y)
            square = image[x:xp, y:yp]
            # print("square", square.shape)
            cv2.circle(draw, (x, y), 5, (255, 0, 0), -1)

            if config[i][j] == 0 and not is_black_square(square):
                return False

            if config[i][j] == 1 and not is_white_square(square):
                return False

            x += dx
            y += dy

    return True


def is_corner_square(image, draw, box):
    a, b, c, d = box

    u = (b - a) / SIZE_RATIO
    v = (d - a) / SIZE_RATIO

    sx, sy = a
    dx = u[0] + v[0]
    dy = u[1] + v[1]

    print(dx, dy)

    for i in range(SIZE_RATIO):
        for j in range(SIZE_RATIO):

            x = int(sx + i*dx)
            y = int(sy + j*dy)

            xp = int(sx + (i+1)*dx)
            yp = int(sy + (j+1)*dy)

            if xp < x:
                [x, xp] = [xp, x]

            if yp < y:
                [y, yp] = [yp, y]

            print("center", x, y)
            square = image[x:xp, y:yp]
            # print("square", square.shape)
            cv2.circle(draw, (x, y), 5, (255, 0, 0), -1)

            if i == 2 and j == 2:
                if not is_black_square(square):
                    return False
            else:
                if not is_white_square(square):
                    print("==========================")
                    print("==========================")
                    return False

            x += dx
            y += dy

    return True


# def is_corner_square(image, box):
#     a, b, c, d = box

#     u = (b - a) / SIZE_RATIO
#     v = (d - a) / SIZE_RATIO

#     # norm_u = np.linalg.norm(u)
#     # norm_v = np.linalg.norm(v)
    
#     # u = u / norm_u
#     # v = v / norm_v

#     for i in range(SIZE_RATIO):
#         for j in range(SIZE_RATIO):
#             center = a + 0.5*u + 0.5*v + u*i + v*j
#             center = (int(center[0]), int(center[1]))
#             print(center)
#             cv2.circle(image, center, 5, (255, 0, 0), -1)



image = cv2.imread("qr.png")

black_lower = (0, 0, 0)
black_upper = (180, 255, 30)

blurred = cv2.GaussianBlur(image, (11, 11), 0)
hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

mask = cv2.inRange(hsv, black_lower, black_upper)
# mask = cv2.erode(mask, None, iterations=2)
# mask = cv2.dilate(mask, None, iterations=2)

print(mask.shape)
print(mask.dtype)

# print(mask[80:100, 80:100])

contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_NONE)

image = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
# cv2.drawContours(image, contours, -1, (0,255,0), 3)

for i, cnt in enumerate(contours):
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    if i == 2 or i == 6 or i == 7:
        # cv2.drawContours(image, [box] ,0, (0, 0, 255), 2)
        if is_corner_square(mask, image, box):
            cv2.drawContours(image, [box] ,0, (0, 0, 255), 2)

        if is_main_square(mask, image, box):
            cv2.drawContours(image, [box] ,0, (0, 0, 255), 2)

    # print(box)
    # print()
    # is_corner_square(image, box)

# image = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
cv2.imshow("frame", image)
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break