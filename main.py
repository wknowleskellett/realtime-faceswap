import numpy as np
import cv2 as cv
import contextlib

@contextlib.contextmanager
def video_capture_wrapper(*args, **kwargs):
    cap = cv.VideoCapture(*args, **kwargs)
    if not cap.isOpened():
        print('opening failed')
        return
    try:
        yield cap
    finally:
        cap.release()

def string_ord(num):
    ret_string = ""
    while num != 0:
        ret_string += chr(num & 0xFF)
        num //= 0x100
    return ret_string

channels = (True, True, True)
width, height = 1920, 1080
with video_capture_wrapper(0, cv.CAP_DSHOW) as cap:
    print(cap.getBackendName())
    cap.set(cv.CAP_PROP_FPS, 30.0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
    print(cap.get(cv.CAP_PROP_FRAME_WIDTH), cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    print(cap.get(cv.CAP_PROP_FPS))
    print(string_ord(int(cap.get(cv.CAP_PROP_FOURCC))))
    i_did_the_thing = False
    while cap.isOpened():
        ret, frame = cap.read()
        swidth, sheight = int(width*0.4), int(height*0.4)
        frame = cv.flip(frame, 1)
        smol = cv.resize(frame, dsize=(swidth, sheight), interpolation=cv.INTER_CUBIC)
        
        for i, has_channel in enumerate(channels):
            if not has_channel:
                smol[:,:,i] = 0
        if not i_did_the_thing:
            print(type(smol))
            i_did_the_thing = True

        people = (smol[:,swidth//2:], smol[:,:swidth//2])
        top_row = np.concatenate(people, axis=1)
        swap = people[1], people[0]
        bottom_row = np.concatenate(swap, axis=1)
        vis = np.concatenate((top_row, bottom_row), axis=0)
##        cv.imshow('Webcam', smol)
        cv.imshow('Webcam', vis)
        key = chr(cv.waitKey(1) & 0xFF)
        if not ret or key == 'q':
            break
        elif key == 'b':
            channels = (not channels[0], channels[1],     channels[2])
        elif key == 'g':
            channels = (channels[0],     not channels[1], channels[2])
        elif key == 'r':
            channels = (channels[0],     channels[1],     not channels[2])
cv.destroyAllWindows()
