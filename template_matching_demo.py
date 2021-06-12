import logging
import timeit

import cv2

import pysift

logger = logging.getLogger(__name__)

MIN_MATCH_COUNT = 10

img1 = cv2.imread('box.png', 0)  # queryImage

start = timeit.default_timer()
kp1, des1 = pysift.computeKeypointsAndDescriptors(img1)
stop = timeit.default_timer()
print(kp1)
print(des1)
print('Time: ', stop - start)
