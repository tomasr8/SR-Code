import time
import cv2
from qr.read_qr import read_qr

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('annina.mp4')
# cap = cv2.VideoCapture(0)


cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # set new dimensionns to cam object (not cap)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error opening video stream or file")

# Read until video is completed
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret:

        start = time.time()
        res = read_qr(frame)
        if res.qr_code_visualization is not None:
            vis = cv2.resize(res.qr_code_visualization, (300, 300))
            res.contour_visualization[:300, :300] = vis
        # frame[:300, :300] = vis

        cv2.imshow('Frame', res.contour_visualization)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()
