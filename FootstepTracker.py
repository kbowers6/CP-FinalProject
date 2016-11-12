import numpy as np
import cv2

import os
import random
import PIL.Image
import matplotlib.pyplot as plt
import math

# CD = ||I - aE||
# Orthogonal distance from aE to I
def ComputeChromacityDistortion(foregroundImage, backgroundImage, x, y, brightnessDistortion):
    if brightnessDistortion == 0:
        return 0
    else:
        I = [float(i) for i in foregroundImage[y][x]]
        E = [float(i) for i in backgroundImage[y][x]]
        aE = np.multiply(brightnessDistortion, E)
        CDvector = np.subtract(I, aE)
        CD = math.sqrt(np.vdot(CDvector,CDvector))
        return CD


# (I DOT E) / ||E||^2 = BD (or a)
def ComputeBrightnessDistortion(foregroundImage, backgroundImage, x, y):
    I = [float(i) for i in foregroundImage[y][x]]
    E = [float(i) for i in backgroundImage[y][x]]
    Emag2 = np.vdot(E, E)
    if Emag2 == 0:
        return 0
    else:
        BD = np.vdot(I, E) / Emag2
        return BD

# Filters out shadow and replaces it with the background pixels
def ShadowFilter(foregroundImage, backgroundImage, CDthreshold=30.0, BDthreshold=0.8):
    for rowIter in range(foregroundImage.shape[0]):
        for colIter in range(foregroundImage.shape[1]):
            BD = ComputeBrightnessDistortion(foregroundImage, backgroundImage, colIter, rowIter)
            CD = ComputeChromacityDistortion(foregroundImage, backgroundImage, colIter, rowIter, BD)

            if CD < CDthreshold:
                if BDthreshold < BD < 1.0: # > 1.0 is a highlight
                    # Shadow
                    foregroundImage[rowIter][colIter][0] = backgroundImage[rowIter][colIter][0]
                    foregroundImage[rowIter][colIter][1] = backgroundImage[rowIter][colIter][1]
                    foregroundImage[rowIter][colIter][2] = backgroundImage[rowIter][colIter][2]


# Look along the line segment (startingpt -> endingPt) for 2 positive streaks => feet
def DetectFeetContour(thresh, startingPt, endingPt, footOffset):
    #iterate along the line segment
    lineSegmentLength = endingPt[0] - startingPt[0]

    xStarting = startingPt[0]
    yIntersect = endingPt[1] - footOffset

    prevPixel = False
    numToggles = 0
    for iter in range(lineSegmentLength):
        if not prevPixel and thresh[yIntersect][xStarting + iter] > 0:
            prevPixel = True
            numToggles += 1
        elif prevPixel and thresh[yIntersect][xStarting + iter] ==  0:
            prevPixel = False
    return numToggles

cap = cv2.VideoCapture("1.avi")

# initialize the first frame in the video stream
firstFrame = None
firstFrameColor = None

kernelSize = 5
threshold = 20
dilateIterations = 6
footOffsetPercentage = 0.10
minArea = 2000
skipToFrame = 50
frameCount = 0
toggleShadowFilter = False
toggleRatioFilter = True
ratio = 2.0 # Ratio height : width (vertical rectangle
while(1):
    frameCount += 1
    print 'Frame: ', frameCount

    ret, frame = cap.read()

    if frame is None:
        break

    # Detect Shadow
    if toggleShadowFilter and firstFrame is not None and frameCount >= skipToFrame:
        ShadowFilter(frame, firstFrameColor)

    # resize the frame, convert it to grayscale, and blur it
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (kernelSize, kernelSize), 0)

    # if the first frame is None, initialize it
    if firstFrame is None:
        firstFrame = gray
        firstFrameColor = frame.copy()
        continue
    if frameCount < skipToFrame:
        continue

    # compute the absolute difference between the current frame and
    # first frame
    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, threshold, 255, cv2.THRESH_BINARY)[1]

    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=dilateIterations)
    im2, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # loop over the contours
    maxContourAreaIdx = -1
    maxContourArea = 0
    idx = 0
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)

        # Filter out any non-vertical-rectangular contours
        if toggleRatioFilter:
            if float(h) / float(w) < ratio:
                print 'Filtered ', float(h)/float(w), h, w
                idx += 1
                continue

        if cv2.contourArea(c) > maxContourArea:
            maxContourArea = cv2.contourArea(c)
            maxContourAreaIdx = idx

        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        idx += 1

    # The largest Contour
    if maxContourAreaIdx > -1 and maxContourArea > minArea:
        (x, y, w, h) = cv2.boundingRect(contours[maxContourAreaIdx])
        print 'largest w,h = ', w,h
        footOffset = int(h * footOffsetPercentage)
        footCount = DetectFeetContour(thresh, (x, y), (x + w, y + h), footOffset)

        # largest contour bounding box and estimated footline
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.line(frame,(x, y+h-footOffset), (x+w, y+h-footOffset), (0,255,0), 2)

        cv2.rectangle(thresh, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.line(thresh, (x, y + h - footOffset), (x + w, y + h - footOffset), (0, 255, 0), 2)

        # Outline the feet
        if footCount == 2:
            cv2.rectangle(frame, (x, y+h-footOffset - 10), (x + w, y+h-footOffset + 10), (255, 0, 255), 2)

    # cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

    # show the frame and record if the user presses a key
    # cv2.imshow("Frame Delta", frameDelta)
    cv2.imshow("Thresh", thresh)
    cv2.imshow('frame', frame)
    key = cv2.waitKey() & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

































