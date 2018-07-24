#
# Copyright (c) 2017, Manfred Constapel
# This file is licensed under the terms of the MIT license.
#


import cv2 as cv
import numpy as np
import PIL.Image


def sobel(img):
    """ just for fun: (Sobel of x) XOR (Sobel of y) """
    framex = cv.Sobel(img, cv.CV_8U, 1, 0)
    datax = np.array(framex, dtype=np.uint8)
    framey = cv.Sobel(img, cv.CV_8U, 0, 1)
    datay = np.array(framey, dtype=np.uint8)
    result = np.where((datax > datay), datax, datay).astype('uint8')
    return np.asarray(result, dtype=np.uint8),


def canny(img, width=0.1):
    """ adaptive Canny filter, edge detector  """
    avg = np.average(img)  # or median?
    std = int(np.std(img) * width)
    lower = int(max(0, avg - std))
    upper = int(min(255, avg + std))
    edges = cv.Canny(img, lower, upper, apertureSize = 3)
    return edges,


def masking(img, low=32, high=224):
    """ masking by threshold (b/w) """
    lower = np.array(low)
    upper = np.array(high)
    mask = cv.inRange(img, lower, upper)  # 1 = white, 0 = black
    frame = img * mask
    return frame, 
    #kernel = np.ones((1, 1), np.uint8)  # convolution matrix
    #return cv.dilate(mask, kernel, iterations=1)


def hough(img, min_length=10):
    """ Hough transformation, corner detection """
    frame = np.zeros(img.shape[:2], np.uint8)
    edges, *_ = canny(img)

    lines = cv.HoughLinesP(edges, 1, np.pi / 180, min_length)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                p = ((x1+x2) // 2, (y1+y2) // 2)
                x, y = p
                cv.line(frame, (x, y), (x, y), 255, 1)  # dot

    image = PIL.Image.fromarray(frame)
    frame = np.asarray(image, dtype=np.uint8)
    return frame,
