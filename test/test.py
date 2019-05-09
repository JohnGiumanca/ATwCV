import numpy as np
import argparse
import imutils
import glob
import cv2
import pytesseract as pytess
from PIL import Image

# Notes: cu Canny se face template matching pe edge-uri, de testat
# in get_elements_coordinates se face multiscale pentru elemente, poate dam scara ca parametru


cv2_img = cv2.imread('ex3.png')
cv2_img = cv2.cvtColor(cv2_img,cv2.COLOR_BGR2GRAY)
cv2_img = cv2.resize(cv2_img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
cv2_img = cv2.GaussianBlur(cv2_img, (5, 5), 0)
cv2_img = cv2.threshold(cv2_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
cv2.imshow('img',cv2_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# img = Image.open('ex1.png')
# img.show()
pil_img = Image.fromarray(cv2_img)
string_from_image = pytess.image_to_string(pil_img)
print string_from_image
