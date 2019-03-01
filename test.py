import numpy as np
import argparse
import imutils
import glob
import cv2
import elements as el

cursor = cv2.imread('Assets/cursor3.png')
cursor = cv2.cvtColor(cursor, cv2.COLOR_BGR2GRAY)
cursor1 = cv2.Canny(cursor, 10, 100)
cursor2 = cv2.Canny(cursor, 20, 200)
cursor3 = cv2.Canny(cursor, 40, 300)
cursor4 = cv2.Canny(cursor, 50, 200)
cv2.imshow("img1",cursor1)
cv2.imshow("img2",cursor2)
cv2.imshow("img3",cursor3)
cv2.imshow("img4",cursor4)
cv2.waitKey(0)