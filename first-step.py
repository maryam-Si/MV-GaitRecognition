import cv2 as cv
cap = cv.VideoCapture(0)

# using mobile's camera
url= "http://192.168.43.1:8080/video"
cap.open(url)

# subtracting background

fgbg = cv.createBackgroundSubtractorKNN()


# other ways

#fgbg = cv.bgsegm.createBackgroundSubtractorK()
#fgbg = cv.bgsegm.BackgroundSubtractorGMG()
#fgbg = cv.createBackgroundSubtractorMOG2(detectShadows=True)

while True:
    ret, frame = cap.read()
    if frame is None:
        break
    fgmask = fgbg.apply(frame)

    # removing noise by median filter
    filteredFrame = cv.medianBlur(fgmask,5)

    cv.imshow('Frame', frame)
    cv.imshow('FG MASK Frame', filteredFrame)

    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
    
cap.release()
cv.destroyAllWindows()
