import numpy as np
import argparse
import imutils
import glob
import cv2
import pytesseract as pytess
from PIL import Image

# Notes: cu Canny se face template matching pe edge-uri, de testat
# in get_elements_coordinates se face multiscale pentru elemente, poate dam scara ca parametru
def find_element(image, element, threshold = 0.9, edge_detection = False, multi_scale = False, visualize = False):
	
	if multi_scale:
		scales = np.linspace(0.2, 1.0, 20)[::-1] 
	else:
		scales = [1]			

	(tH, tW) = element.shape[:2]
	found = None

	for scale in scales:

		image_resized = imutils.resize(image, width = int(image.shape[1] * scale))
		r = image.shape[1] / float(image_resized.shape[1])

		if image_resized.shape[0] < tH or image_resized.shape[1] < tW:
			break

		if edge_detection:
			image_resized = cv2.Canny(image_resized,50,200)
			element = cv2.Canny(element,50,200)
		
		result = cv2.matchTemplate(image_resized, element, cv2.TM_CCOEFF_NORMED)
		(_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

		if visualize:
			clone = np.dstack([result, result, result])
			cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
				(maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
			cv2.imshow("Visualize", clone)
			cv2.waitKey(0)

		if found is None or maxVal > found[0]:
			found = (maxVal,maxLoc,r)
	
	(maxVal, maxLoc, r) = found
	if maxVal > threshold:
		(startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
		(endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
	else:
		(startX, startY,endX, endY) = (0,0,0,0)

	return (startX, startY,endX, endY)







sc= cv2.imread('checkbox_field.png')
check = cv2.imread('checkbox_on.png')

(startX, startY,endX, endY) = find_element(sc,check)
while (startX, startY,endX, endY) != (0,0,0,0):
	
	textbox_startX, textbox_startY = endX, startY
	textbox_endX, textbox_endY= endX + 150, endY 
	text_image = sc[textbox_startY:textbox_endY,textbox_startX:textbox_endX]
	cv2.rectangle(sc, (textbox_startX, textbox_startY), (textbox_endX, textbox_endY), (51, 255, 153), 3)
	cv2.imshow('img',sc)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	image = cv2.cvtColor(text_image,cv2.COLOR_BGR2GRAY)
	image = cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
	image = cv2.GaussianBlur(image, (5, 5), 0)
	image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
	text_string = pytess.image_to_string(image, lang='eng')
	print text_string
	sc[startY:endY,startX:endX] = (0,0,0) 
	(startX, startY,endX, endY) = find_element(sc,check)
	

# cv2_img = cv2.cvtColor(cv2_img,cv2.COLOR_BGR2GRAY)
# cv2_img = cv2.resize(cv2_img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
# cv2_img = cv2.GaussianBlur(cv2_img, (5, 5), 0)
# cv2_img = cv2.threshold(cv2_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
# cv2.imshow('img',cv2_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# img = Image.open('ex1.png')
# img.show()
# pil_img = Image.fromarray(cv2_img)
# string_from_image = pytess.image_to_string(pil_img)
# print string_from_image













