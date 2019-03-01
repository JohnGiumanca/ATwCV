import numpy as np
import argparse
import imutils
import glob
import cv2
import elements as el

# Notes: cu Canny se face template matching pe edge-uri, de testat
# in get_elements_coordinates se face multiscale pentru elemente, poate dam scara ca parametru

cursor_path = 'Assets/cursor3.png'
elements_path = 'Assets/elements/'
input_path = 'Assets/app_rec_2.mov'
elements_img_type = 'png'

visualize = False             
tm_threshold = 0.6
intensity_threshold = 10


# Elements load
(elements,elements_id) = el.load_elements(elements_path, elements_img_type)

cursor = cv2.imread(cursor_path)
cursor = cv2.cvtColor(cursor, cv2.COLOR_BGR2GRAY)
# cursor = cv2.Canny(cursor, 50, 200)
(tH, tW) = cursor.shape[:2]

cap = cv2.VideoCapture(input_path)

if cap.isOpened() == False:
	print("Error opening video stream or file!")

_ , screenshot = cap.read()

elements_coord = el.get_elements_coordinates(elements, elements_id, screenshot,tm_threshold)
elements_color_diff = dict()
for eid in elements_id:
	if elements_coord[eid] != None:
		coord_image = screenshot[	elements_coord[eid][0][1]:elements_coord[eid][1][1],
									elements_coord[eid][0][0]:elements_coord[eid][1][0]	]
		avg1 = cv2.mean(elements[eid])[0:3]
		avg2 = cv2.mean(coord_image)[0:3]
		elements_color_diff[eid] = el.color_diff(avg1,avg2) 


click = False

while cap.isOpened():
	ret , image = cap.read()
	if ret == False:
		break
	image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	found = None
	# scales = np.linspace(0.2, 1.0, 20)[::-1] 	#multi scale
	scales = [1]; 								#single scale

	for scale in scales:

		resized = imutils.resize(image_gray, width = int(image_gray.shape[1] * scale))
		r = image_gray.shape[1] / float(resized.shape[1])

		if resized.shape[0] < tH or resized.shape[1] < tW:
			break

		# image_edge = cv2.Canny(resized,50,200)
		result = cv2.matchTemplate(resized, cursor, cv2.TM_CCOEFF_NORMED)
		(_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

		if visualize:
			clone = np.dstack([image_edge, image_edge, image_edge])
			cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
				(maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
			cv2.imshow("Visualize", clone)
			cv2.waitKey(0)

		if found is None or maxVal > found[0]:
			found = (maxVal,maxLoc,r)
	print maxVal	
	(maxVal, maxLoc, r) = found
	if maxVal > tm_threshold:
		(startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
		(endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
	else:
		(startX, startY,endX, endY) = (0,0,0,0)

	# draw a bounding box around the detected result and display the image
	
	cv2.rectangle(image, (startX, startY), (endX, endY), (51, 51, 255), 3)
	for eid in elements_id:
		color = (51, 255, 153)

		
		if elements_coord[eid] != None and el.do_overlap( elements_coord[eid][0], elements_coord[eid][1],
															(startX, startY), (endX, endY)):
			color = (51, 225, 255)
			el_image = image[	elements_coord[eid][0][1]:elements_coord[eid][1][1],
								elements_coord[eid][0][0]:elements_coord[eid][1][0]  ]
			avg1 = cv2.mean(elements[eid])[0:3]
			avg2 = cv2.mean(el_image)[0:3] 
			intensity_diff = abs(elements_color_diff[eid] - el.color_diff(avg1,avg2))
			
			
			if intensity_diff > intensity_threshold:
				click = True

			if intensity_diff < intensity_threshold and click == True:
				# print(abs(elements_color_diff[eid] - el.color_diff(avg1,avg2)))
				print("Element " + str(eid) + " pressed!")
				click = False

			
			
		cv2.rectangle(image, elements_coord[eid][0], elements_coord[eid][1], color, 3)


	resized = imutils.resize(image, width = int(image_gray.shape[1] * 0.4))
	cv2.imshow("Image", resized)
	if cv2.waitKey(25) & 0xFF == ord('q'):
		break           

cap.release()
cv2.destroyAllWindows()



