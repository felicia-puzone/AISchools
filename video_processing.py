import numpy as np
import cv2 as cv

from PaletteGen import palette_extractor
from Downsampling import pixxelate

from prova_gen_resnet import resnet_process


######### Registrazione da webcam #############

"""
cap = cv.VideoCapture(0)
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('output.avi', fourcc, 20.0, (640,  480))
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    out.write(frame)
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break
# Release everything if job is finished
cap.release()
out.release()
cv.destroyAllWindows()
"""

####################################################


################### Pixxelazione video #############

cap = cv.VideoCapture('video_in.mp4')
# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('video_out.avi', fourcc, 20.0, (int(cap.get(3)),int(cap.get(4))))
while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    palette = palette_extractor(frame) 
    frame = resnet_process(frame)
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
