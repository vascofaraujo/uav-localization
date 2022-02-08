import cv2 as cv

video_path = 'video.h264'

cap = cv.VideoCapture(video_path)

while(cap.isOpened()):
    ret, frame = cap.read()

    frame = cv.resize(frame, (600, 300))

    cv.imshow('window', frame)

    cv.waitKey(30)
