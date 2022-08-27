import time

import click
import cv2
import numpy as np

from sr.code import ContourResult, DecodeError, DecodeResult, SRCodeReader, video_capture
from sr.image import draw_contours, find_contours, get_pressed_key, horizontal_concat, make_black_and_white


def decode(image, find_all=True, use_bw=False):
    start_time = time.time()

    bw = make_black_and_white(image)
    contours = find_contours(bw)
    result = DecodeResult(contour_visualization=cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB) if use_bw else image.copy())
    draw_contours(result.contour_visualization, contours)

    for contour in contours:
        sr = SRCodeReader(size=np.sqrt(contour.area), image=image)

        try:
            message = sr.decode(contour)
        except DecodeError as e:
            result.contours.append(ContourResult(error=str(e)))
        else:
            visualization = cv2.cvtColor(sr.image, cv2.COLOR_GRAY2RGB)
            sr.visualize_decoded(visualization)
            result.contours.append(ContourResult(message=message, visualization=visualization))

            if not find_all:
                break

    result.time_ms = (time.time() - start_time) * 1000
    return result


def decode_video(filename, visualize=True, verbose=True, stop_on_success=True):
    with video_capture(filename) as cap:
        cv2.namedWindow('SR code', cv2.WINDOW_KEEPRATIO)
        use_bw = False
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            key = get_pressed_key(timeout_ms=10)
            if key == 'q':
                break

            if key == 'b':
                use_bw = not use_bw
            result = decode(frame, find_all=True, use_bw=use_bw)

            if result.success and stop_on_success:
                click.echo(result)
                break

            if verbose:
                click.echo(result)

            if not visualize:
                continue

            visualization = result.contour_visualization
            if result.success:
                for vis in result.get_visualizations():
                    visualization = horizontal_concat(visualization, vis)
            cv2.imshow('SR code', visualization)

        cv2.destroyAllWindows()
