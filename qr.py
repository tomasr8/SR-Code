import matplotlib.pyplot as plt
import numpy as np
import cv2

from homography import solve_homography


def is_black(selection, threshold=0.6):
    total_area = 1
    for dim in selection.shape:
        total_area *= dim

    black_area = np.sum(selection == 0)

    # print("ratio", black_area/total_area)

    if (black_area/total_area) > threshold:
        return True
    else:
        return False


def is_white(selection, threshold=0.6):
    total_area = 1
    for dim in selection.shape:
        total_area *= dim

    white_area = np.sum(selection == 255)

    # print("ratio", white_area/total_area)

    if (white_area/total_area) > threshold:
        return True
    else:
        return False


def create_circular_mask(x, y, r, w, h):
    Y, X = np.ogrid[:h, :w]

    dist_from_center = np.sqrt((X - x)**2 + (Y - y)**2)

    mask = dist_from_center <= r
    return mask


def is_main_square(image, draw, box):
    H, W = image.shape

    center = np.mean(box, axis=0)
    # print(box)
    # print(center)

    BIG_CIRCLE_RADIUS = 2/22
    SMALL_CIRCLE_RADIUS = 1/22

    A, B, C, D = box
    side_length = np.linalg.norm(B - A)

    big_mask = create_circular_mask(
        center[0], center[1], side_length * BIG_CIRCLE_RADIUS, W, H)
    small_mask = create_circular_mask(
        center[0], center[1], side_length * SMALL_CIRCLE_RADIUS, W, H)

    small_circle = image[small_mask]

    # print(is_black(small_circle, threshold=0.7))
    # draw[small_mask] = [0, 255, 0]

    big_circle = image[big_mask & (~small_mask)]
    # draw[big_mask & (~small_mask)] = [0, 255, 0]

    cv2.circle(draw, (int(center[0]), int(center[1])), int(
        side_length * BIG_CIRCLE_RADIUS), [0, 255, 0])
    cv2.circle(draw, (int(center[0]), int(center[1])), int(
        side_length * SMALL_CIRCLE_RADIUS), [0, 255, 0])

    # print(is_white(big_circle, threshold=0.7))
    return is_black(small_circle, threshold=0.7) and is_white(big_circle, threshold=0.7)


def get_square(image, size, i, j):
    y = int(i*size)
    x = int(j*size)

    yp = int((i+1)*size)
    xp = int((j+1)*size)

    square = image[y:yp, x:xp]
    return square


def get_squares(image, size, yjs):
    squares = []

    for (i, j) in yjs:
        y = int(i*size)
        x = int(j*size)

        yp = int((i+1)*size)
        xp = int((j+1)*size)

        square = image[y:yp, x:xp]
        squares.append(square)

    return squares


def is_start_corner(corner):
    # |XX|--|
    # |--|
    middle, left, right = corner

    if is_white(middle) and is_black(left) and is_black(right):
        return True
    else:
        return False


def is_end_corner(corner):
    # |XX|XX|
    # |XX|
    middle, left, right = corner

    if is_white(middle) and is_white(left) and is_white(right):
        return True
    else:
        return False


def find_corners(image, size):
    start = None
    end = None

    top_left = get_squares(image, size, [
        [1, 1],
        [1, 2],
        [2, 1]
    ])

    bot_left = get_squares(image, size, [
        [20, 1],
        [19, 1],
        [20, 2]
    ])

    top_right = get_squares(image, size, [
        [1, 20],
        [1, 19],
        [2, 20]
    ])

    bot_right = get_squares(image, size, [
        [20, 20],
        [20, 19],
        [19, 20]
    ])

    if is_start_corner(top_left):
        start = np.array([2, 2])
    elif is_end_corner(top_left):
        end = np.array([2, 2])

    if is_start_corner(bot_left):
        start = np.array([19, 2])
    elif is_end_corner(bot_left):
        end = np.array([19, 2])

    if is_start_corner(top_right):
        start = np.array([2, 19])
    elif is_end_corner(top_right):
        end = np.array([2, 19])

    if is_start_corner(bot_right):
        start = np.array([19, 19])
    elif is_end_corner(bot_right):
        end = np.array([19, 19])

    return start, end


def read_data(image, draw, size):
    start, end = find_corners(image, size)

    # print("start", start)
    # print("end", end)
    # print("====")

    dy, dx = np.sign(end - start)
    # print(dy)
    # print(dx)

    assert ((dx == 0) or (dy == 0))

    if dy == 0:
        vertical = True
        if start[0] == 2:
            dy = 1
        else:
            dy = -1
    else:
        vertical = False
        if start[1] == 2:
            dx = 1
        else:
            dx = -1

    # print(dy, dx)

    data = []

    if vertical:
        k = 0
        for i in range(18):
            for j in range(18):
                y = start[0] + dy*i
                x = start[1] + dx*j

                if (y >= 8 and y <= 13) and (x >= 8 and x <= 13):
                    continue

                square = get_square(image, size, y, x)
                if is_white(square, threshold=0.5):
                    data.append(0)
                    # cv2.putText(draw, f"{k}", (int(x*size + size/2), int(y*size + size/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [0, 200, 50])
                    cv2.circle(draw, (int(x*size + size/2), int(y*size + size/2)), 6, [100, 100, 0], -1)
                else:
                    data.append(1)
                    # cv2.putText(draw, f"{k}", (int(x*size + size/2), int(y*size + size/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [0, 200, 50])
                    cv2.circle(draw, (int(x*size + size/2), int(y*size + size/2)), 6, [100, 0, 100], -1)

                k += 1

    else:
        for j in range(18):
            for i in range(18):
                y = start[0] + dy*i
                x = start[1] + dx*j

                if (y >= 8 and y <= 13) and (x >= 8 and x <= 13):
                    continue

                square = get_square(image, size, y, x)
                if is_white(square, threshold=0.5):
                    data.append(0)
                    # cv2.putText(draw, "1", (int(x*size + size/2), int(y*size + size/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, [0, 200, 50])
                    cv2.circle(draw, (int(x*size + size/2), int(y*size + size/2)), 6, [100, 100, 0], -1)
                else:
                    data.append(1)
                    # cv2.putText(draw, "0", (int(x*size + size/2), int(y*size + size/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, [0, 200, 50])
                    cv2.circle(draw, (int(x*size + size/2), int(y*size + size/2)), 6, [100, 0, 100], -1)

    # print(data)
    decoded = decode_data(data)
    print(decoded)
    return decoded


def decode_data(data):
    n = len(data)//3

    data = np.reshape(data, (3, n))

    data = np.sum(data, axis=0)

    data[data <= 1] = 0
    data[data >= 2] = 1

    n = len(data)//6
    data = np.reshape(data, (n, 6))

    BASE64 = " !0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

    data = [binary_array_to_num(arr) for arr in data]
    data = [BASE64[i] for i in data]
    return "".join(data)


def binary_array_to_num(arr):
    return int("".join(str(x) for x in arr), 2)


def read_qr(path):
    image = cv2.imread(path)
    image = cv2.resize(image, (1280, 960), interpolation=cv2.INTER_AREA)

    black_lower = (0, 0, 0)
    black_upper = (180, 255, 70)

    # blurred = cv2.GaussianBlur(image, (11, 11), 0)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, black_lower, black_upper)
    # mask = cv2.erode(mask, None, iterations=2)
    # mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

    cv2.imshow("frame", image)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    for i, cnt in enumerate(contours):
        # print(cv2.contourArea(cnt))
        # rect = cv2.minAreaRect(cnt)
        # box = cv2.boxPoints(rect)
        # box = np.int0(box)

        if (cv2.contourArea(cnt) < 100):
            continue

        perimeter = cv2.arcLength(cnt, True)
        epsilon = 0.05*cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if not cv2.isContourConvex(approx):
            continue

        if approx.shape[0] == 4:
            # cv2.drawContours(image, [approx], -1, (255, 0, 50), 2)

            rect = approx[:, 0, :]
            target = np.float64([
                [0, 0],
                [0, 880],
                [880, 880],
                [880, 0]
            ])

            H = solve_homography(rect, target)

            image_h = cv2.warpPerspective(image, H, (1280, 960))
            image_h = image_h[:881, :881]

            mask_h = cv2.warpPerspective(mask, H, (1280, 960))
            mask_h = mask_h[:881, :881]

            # is_main_square(mask_h, image_h, target)
            if not is_main_square(mask_h, image_h, target):
                continue

            size = 880/22
            for i in range(22):
                for j in range(22):
                    cv2.circle(image_h, (int(size*i), int(size*j)),
                               3, (100, 20, 200), -1)

            read_data(mask_h, image_h, size)

            cv2.imshow("frame", image_h)
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break


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
    read_qr("qr5_slanted2.jpg")
