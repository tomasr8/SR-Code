import time

import click
import cv2
import numpy as np

from sr.code import ContourResult, DecodeError, DecodeResult, SRCodeReader, video_capture
from sr.image import (draw_contours, find_contours, horizontal_concat,
                      make_black_and_white, pressed_quit)


def decode(image, find_all):
    start_time = time.time()

    bw = make_black_and_white(image)
    contours = find_contours(bw)
    result = DecodeResult(contour_visualization=cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB))
    draw_contours(result.contour_visualization, contours)

    for contour in contours:
        sr = SRCodeReader(size=np.sqrt(contour.area), image=bw)

        try:
            message = sr.decode(contour)
        except DecodeError as e:
            result.contours.append(ContourResult(error=str(e)))
        else:
            visualization = cv2.cvtColor(sr.image, cv2.COLOR_GRAY2RGB)
            sr.visualize_decoded(visualization)
            result.contours.append(ContourResult(message=message, visualization=visualization))

    result.time_ms = (time.time() - start_time) * 1000
    return result


def decode_video(filename, visualize, verbose, stop_on_success):
    with video_capture(filename) as cap:
        cv2.namedWindow('SR code', cv2.WINDOW_KEEPRATIO)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            result = decode(frame)

            if result.success and stop_on_success:
                click.echo(result)
                break

            if verbose:
                click.echo(result)

            if not visualize:
                continue

            if result.message:
                visualization = horizontal_concat(result.contour_visualization, result.decode_visualization)
            else:
                visualization = result.contour_visualization

            cv2.imshow('SR code', visualization)
            if pressed_quit(timeout_ms=10):
                break

        cv2.destroyAllWindows()
