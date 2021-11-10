import time
import cv2
import numpy as np
from qr import read_qr_video

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('qr5_video.mp4')

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:

    print("shape", frame.shape)
    # frame = cv2.resize(frame, (1920, 960), interpolation=cv2.INTER_AREA)

    # Display the resulting frame
    start = time.time()
    frame = read_qr_video(frame)
    frame = cv2.resize(frame, (960, 480), interpolation=cv2.INTER_AREA)
    cv2.imshow('Frame', frame)
    print("time", (time.time() - start))

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

  # Break the loop
  else: 
    break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()

