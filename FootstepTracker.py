import numpy as np
import cv2

import os
import random
import PIL.Image
import matplotlib.pyplot as plt
import math
from enum import Enum

class Direction(Enum):
    Left = 'Left'
    Right = 'Right'
    Unsure = 'Unsure'

# CD = ||I - aE||
# Orthogonal distance from aE to I
def ComputeChromacityDistortion(foregroundImage, backgroundImage, brightnessDistortion):
    # Convert to 3 channel since we are scaling a 3 channel RGB matrix
    BD = np.zeros(foregroundImage.shape)
    BD[:,:,0] = brightnessDistortion
    BD[:,:,1] = brightnessDistortion
    BD[:,:,2] = brightnessDistortion

    aE = np.multiply(BD, backgroundImage)

    CDvector = np.subtract(foregroundImage, aE).astype(float)
    CDvector = CDvector.reshape((foregroundImage.shape[0] * backgroundImage.shape[1], 3))

    CDdot = CDvector * CDvector
    CDdot = np.sum(CDdot, axis=1)
    CD = np.sqrt(CDdot)

    CD = CD.reshape((foregroundImage.shape[0], backgroundImage.shape[1]))

    return CD

# (I DOT E) / ||E||^2 = BD (or a)
def ComputeBrightnessDistortion(foregroundImage, backgroundImage):
    foregroundImageFloat = foregroundImage.astype(float)
    backgroundImageFloat = backgroundImage.astype(float)

    foregroundImageFlattened = foregroundImageFloat.reshape((foregroundImage.shape[0] * backgroundImage.shape[1], 3))
    backgroundImageFlattened = backgroundImageFloat.reshape((backgroundImage.shape[0] * backgroundImage.shape[1], 3))

    BDnum = foregroundImageFlattened * backgroundImageFlattened
    BDnum = np.sum(BDnum, axis=1)

    BDdenom = backgroundImageFlattened * backgroundImageFlattened
    BDdenom = np.sum(BDdenom, axis=1)

    with np.errstate(invalid='ignore'):
        BDmatrix = np.nan_to_num(np.divide(BDnum, BDdenom))

    BDmatrix = BDmatrix.reshape((foregroundImage.shape[0], backgroundImage.shape[1]))

    return BDmatrix

# Filters out shadow and replaces it with the background pixels
def ShadowFilter(foregroundImage, backgroundImage, CDthreshold=30.0, BDthreshold=0.8):
    BD = ComputeBrightnessDistortion(foregroundImage,backgroundImage)
    CD = ComputeChromacityDistortion(foregroundImage, backgroundImage, BD)

    for rowIter in range(foregroundImage.shape[0]):
        for colIter in range(foregroundImage.shape[1]):
            if CD[rowIter,colIter] < CDthreshold:
                if BDthreshold < BD[rowIter,colIter] < 1.0: # > 1.0 is a highlight
                    # Shadow
                    foregroundImage[rowIter][colIter][0] = backgroundImage[rowIter][colIter][0]
                    foregroundImage[rowIter][colIter][1] = backgroundImage[rowIter][colIter][1]
                    foregroundImage[rowIter][colIter][2] = backgroundImage[rowIter][colIter][2]

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
# Look along the line segment (startingpt -> endingPt) for 2 positive streaks => feet

# Computes the direction of the bounding box.
def ComputeDirection(prevUL, prevUR, currUL, currUR):
    if prevUL is None or prevUR is None:
        return Direction.Unsure

    diffULx = prevUL[0] - currUL[0]
    diffURx = prevUR[0] - currUR[0]

    # Moving right
    if diffULx < 0 and diffURx < 0:
        return Direction.Right
    # Moving Left
    elif diffULx > 0 and diffURx > 0:
        return Direction.Left
    # Unsure
    else:
        return Direction.Unsure

def DetectBackEdgePause(prevCorner, currCorner, threshold):
    if -threshold < prevCorner[0] - currCorner[0] < threshold:
        return True
    else:
        return False

cap = cv2.VideoCapture("2.6.avi")

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output2.6.avi', fourcc, 29.0, (720,480))

# initialize the first frame in the video stream
firstFrame = None
firstFrameColor = None

kernelSize = 5
threshold = 20
dilateIterations = 6
footOffsetPercentage = 0.10
footBoundingBoxPercentage = 0.07
minArea = 2000
skipToFrame = 0
frameCount = 0
toggleShadowFilter = True
toggleRatioFilter = True
ratio = 1.7 # Ratio height : width (vertical rectangle

prevUL = None # (x,y)
prevUR = None # (x,y)
prevDirection = None

pauseFrameCounterThreshold = 2
pauseFrameCounter = 0
pausePixelThreshold = 5

backFootPercentage = 0.33

footstepRoIs = []

while(1):
    frameCount += 1
    print 'Frame: ', frameCount

    ret, frame = cap.read()

    if frame is None:
        break

    cleanFrame = frame.copy()

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
        footBoundingBoxOffset = int(h * footBoundingBoxPercentage)
        footCount = DetectFeetContour(thresh, (x, y), (x + w, y + h), footOffset)

        # largest contour bounding box and estimated footline
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.line(frame,(x, y+h-footOffset), (x+w, y+h-footOffset), (0,255,0), 2)

        cv2.rectangle(thresh, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.line(thresh, (x, y + h - footOffset), (x + w, y + h - footOffset), (0, 255, 0), 2)

        # Outline the feet
        if footCount == 2:
            cv2.rectangle(frame, (x, y+h-footOffset - footBoundingBoxOffset), (x + w, y+h-footOffset + footBoundingBoxOffset), (255, 0, 255), 2)

        direction = ComputeDirection(prevUL, prevUR, (x, y), (x + w, y))
        if direction is not Direction.Unsure:
            prevDirection = direction

        if prevDirection == Direction.Left:
            cv2.line(frame, (x + w, y), (x + w, y + h), (255, 255, 0), 2)
            if DetectBackEdgePause(prevUR, (x + w, y), pausePixelThreshold):
                pauseFrameCounter += 1
                if pauseFrameCounter >= pauseFrameCounterThreshold:
                    # Mark the backedge
                    cv2.line(frame, (x + w, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                if pauseFrameCounter >= pauseFrameCounterThreshold:
                    if direction == Direction.Right:
                        footBBUL = (x, y + h - footOffset - footBoundingBoxOffset)  # (x,y)
                        footBBw = w * backFootPercentage
                        footBBh = 2 * footBoundingBoxOffset
                        footstepRoIs.append([footBBUL[0], footBBUL[1], int(footBBw), footBBh])
                    elif direction == Direction.Left:
                        footBBUL = (x + int((1 - backFootPercentage) * w), y + h - footOffset - footBoundingBoxOffset)  # (x,y)
                        footBBw = w * backFootPercentage
                        footBBh = 2 * footBoundingBoxOffset
                        footstepRoIs.append([footBBUL[0], footBBUL[1], int(footBBw), footBBh])

                pauseFrameCounter = 0
        elif prevDirection == Direction.Right:
            cv2.line(frame, (x, y), (x, y + h), (255, 255, 0), 2)
            if DetectBackEdgePause(prevUL, (x, y), pausePixelThreshold):
                pauseFrameCounter += 1
                if pauseFrameCounter >= pauseFrameCounterThreshold:
                    # Mark the backedge
                    cv2.line(frame, (x, y), (x, y + h), (0, 255, 0), 2)
            else:
                if pauseFrameCounter >= pauseFrameCounterThreshold:
                    footBBUL = (x, y + h - footOffset - footBoundingBoxOffset)  # (x,y)
                    footBBw = w
                    footBBh = 2 * footBoundingBoxOffset
                    footstepRoIs.append([footBBUL[0], footBBUL[1], int(footBBw * backFootPercentage), footBBh])

                pauseFrameCounter = 0

        # Set as the previous for the next frame
        prevUL = (x, y)
        prevUR = (x + w, y)

    # cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

    if prevDirection is not Direction.Unsure:
        print "Direction = ", prevDirection

    # Draw past detected foot steps
    if len(footstepRoIs) > 0:
        for RoI in footstepRoIs:
            x, y, w, h = RoI
            cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)

    # show the frame and record if the user presses a key
    # cv2.imshow("Frame Delta", frameDelta)
    cv2.imshow("Thresh", thresh)
    cv2.imshow('frame', frame)

    # write the video frame
    out.write(frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

































