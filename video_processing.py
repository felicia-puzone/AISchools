import numpy as np
import cv2 as cv

from PaletteGen import palette_extractor
from GridGenerator import pixxelate

cap = cv.VideoCapture('video.mp4')
# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('output.avi', fourcc, 10.0, (int(cap.get(3)),int(cap.get(4))))
while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    palette = palette_extractor(frame) 
    frame = pixxelate(frame, 128, palette)
            
            # write the flipped frame
    out.write(frame)
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break


# Release everything if job is finished
cap.release()
out.release()
cv.destroyAllWindows()