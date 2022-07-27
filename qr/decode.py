import time

import cv2
import numpy as np

from qr.code import DecodeError, DecodeResult, SRCodeReader, video_capture
from qr.image import draw_contours, find_contours, make_black_and_white


def decode(image):
    start_time = time.time()
    denoised = cv2.medianBlur(image, 3)
    bw = make_black_and_white(denoised)

    contours = find_contours(bw)
    result = DecodeResult(contour_count=len(contours),
                          contour_visualization=cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB))
    draw_contours(result.contour_visualization, [c.points for c in contours])

    for contour in contours:
        sr = SRCodeReader(size=np.sqrt(contour.area), image=bw)

        try:
            result.message = sr.decode(contour)
            # raise DecodeError("Test error pelase ignore")
        except DecodeError as e:
            result.errors.append(str(e))
        else:
            result.decode_visualization = cv2.cvtColor(sr.image, cv2.COLOR_GRAY2RGB)
            sr.visualize_decoded(result.decode_visualization)

    result.time = time.time() - start_time
    return result


def decode_video(file, return_on_read=False):
    with video_capture(file) as cap:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
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
