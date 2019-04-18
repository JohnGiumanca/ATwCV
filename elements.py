import numpy as np
import argparse
import imutils
import glob
import cv2



	

def check_keyframe(frame, old_frame, threshold):
	diff = cv2.absdiff(frame, old_frame)
	non_zeros = np.count_nonzero(diff > 3)   
	if non_zeros > threshold:
		return True
	else:
		return False


# 	# result = cv2.matchTemplate(frame, current_page, cv2.TM_CCOEFF_NORMED)
	# (_, maxVal, _, _) = cv2.minMaxLoc(result)
	# print(maxVal)
	# if maxVal > threshold:
	# 	return False
	# else:
	# 	return True

def load_pages(file_name):
	pages = dict()

	with open(file_name, "r") as string_file:
		for line in string_file:
			string_list = [s.replace('\n', '') for s in line.split(' ')]
			pages[string_list[0]] = string_list[1:]

	return pages

def get_current_page(elements_coord, pages):
	all_pages = set()

	for page in pages.values():
		all_pages = all_pages.union(page)

	for eid in elements_coord.keys(): 
		if elements_coord[eid] != [(0,0),(0,0)]:
			all_pages = all_pages.intersection(pages[eid])

	if len(all_pages) == 1:
		current_page = all_pages.pop()
		return current_page
	else:
		print("There are more than 1 page with the same elements")
	



def load_elements(path, type):
	elements = dict()
	
	for file in glob.glob(path + '*' + type):					#ex: 'Assets/elements/*png'
		id = file[file.rfind('/') + 1 : file.find('.'+type)]
		elements[id] = cv2.imread(file)
	
	return elements

def get_elements_color_diff(elements, elements_coord, screenshot):
	elements_color_diff = dict()
	
	for eid in elements.keys():
		if elements_coord[eid] != None:
			coord_image = screenshot[	elements_coord[eid][0][1]:elements_coord[eid][1][1],
										elements_coord[eid][0][0]:elements_coord[eid][1][0]	]
			avg1 = cv2.mean(elements[eid])[0:3]
			avg2 = cv2.mean(coord_image)[0:3]
			elements_color_diff[eid] = color_diff(avg1,avg2) 
	
	return elements_color_diff

def get_elements_coordinates(elements, screenshot, threshold):

	screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

	elements_coord = dict()

	for eid in elements.keys(): 
		
		template = cv2.cvtColor(elements[eid], cv2.COLOR_BGR2GRAY)
		(startX, startY,endX, endY) = find_element(screenshot_gray, template, threshold)
		elements_coord[eid]= [(startX, startY),(endX, endY)] 
		
	return elements_coord

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
			clone = np.dstack([image_edge, image_edge, image_edge])
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


def do_overlap((startX1, startY1),(endX1, endY1),(startX2, startY2),(endX2, endY2)):
	if startX1 > endX2 or startX2 > endX1 or startY1 > endY2 or startY2 > endY1:
		return False

	return True

def color_diff((r1,g1,b1),(r2,g2,b2)):
	return np.sqrt((r2-r1)**2 + (g2-g1)**2 + (b2-b1)**2)
         


