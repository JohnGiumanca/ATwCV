import numpy as np
import argparse
import imutils
import glob
import cv2

def load_elements(path, type):
	elements = dict()
	elements_id = [];
	
	for file in glob.glob(path + '*' + type):					#ex: 'Assets/elements/*png'
		id = file[file.rfind('/') + 1 : file.find('.'+type)]
		elements[id] = cv2.imread(file)
		elements_id.append(id)

	return (elements,elements_id)

def get_elements_coordinates(elements, elements_id, screenshot, threshold):

	screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

	elements_coord = dict()

	for eid in elements_id: 
		template = cv2.cvtColor(elements[eid], cv2.COLOR_BGR2GRAY)
		# template = cv2.Canny(template, 50, 200)
		(tH, tW) = template.shape[:2]

		
		found = None
		# scales = np.linspace(0.2, 1.0, 20)[::-1] 	#multi scale
		scales = [1]; 								#single scale
		for scale in scales:

			resized = imutils.resize(screenshot_gray, width = int(screenshot_gray.shape[1] * scale))
			r = screenshot_gray.shape[1] / float(resized.shape[1])

			if resized.shape[0] < tH or resized.shape[1] < tW:
				break

			# image_edge = cv2.Canny(resized,50,200)
			result = cv2.matchTemplate(resized, template, cv2.TM_CCOEFF_NORMED)
			(_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

			if found is None or maxVal > found[0]:
				found = (maxVal,maxLoc,r)

		(maxVal, maxLoc, r) = found
		print maxVal
		if maxVal > threshold:
			(startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
			(endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
		else:
			(startX, startY,endX, endY) = (0,0,0,0)


		elements_coord[eid]= [(startX, startY),(endX, endY)] 
		
	return elements_coord

def do_overlap((startX1, startY1),(endX1, endY1),(startX2, startY2),(endX2, endY2)):
	if startX1 > endX2 or startX2 > endX1 or startY1 > endY2 or startY2 > endY1:
		return False

	return True

def color_diff((r1,g1,b1),(r2,g2,b2)):
	return np.sqrt((r2-r1)**2 + (g2-g1)**2 + (b2-b1)**2)
         


