


import numpy as np
import argparse
import imutils
import glob
import cv2
import elements as el
from pytesseract import image_to_string

# Notes: cu Canny se face template matching pe edge-uri, de testat
# in get_elements_coordinates se face multiscale pentru elemente, poate dam scara ca parametru

cursor_path = 'Assets/cursor3.png'
elements_path = 'Assets/elements/'
input_path = 'Assets/app_rec_3.mov'
elements_img_type = 'png'

visualize = False             
tm_threshold_cursor = 0.6
tm_threshold_elements = 0.8
intensity_threshold = 5


# Elements load
(elements,elements_id) = el.load_elements(elements_path, elements_img_type)

cursor = cv2.imread(cursor_path)
cursor = cv2.cvtColor(cursor, cv2.COLOR_BGR2GRAY)
# cursor = cv2.Canny(cursor, 50, 200)

cap = cv2.VideoCapture(input_path)

if cap.isOpened() == False:
	print("Error opening video stream or file!")

_ , screenshot = cap.read()

elements_coord = el.get_elements_coordinates(elements, elements_id, screenshot,tm_threshold_elements)
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

	(startX, startY,endX, endY) = el.find_element(image_gray, cursor, tm_threshold_cursor)

	# draw a bounding box around the detected result and display the image
	cv2.rectangle(image, (startX, startY), (endX, endY), (51, 51, 255), 3)
	eid = 'loginButton'
	
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
			
			print("Element " + str(eid) + " pressed!")
			click = False

			el_image = image[	elements_coord['emailTextField'][0][1]:elements_coord['emailTextField'][1][1],
								elements_coord['emailTextField'][0][0]:elements_coord['emailTextField'][1][0]  ]

			if(el_image.shape[0] != 0):
				config = ("-l eng --oem 1 --psm 7")
				text = image_to_string(el_image, config=config)
				print(text)

			el_image = image[	elements_coord['passworTextField'][0][1]:elements_coord['passworTextField'][1][1],
								elements_coord['passworTextField'][0][0]:elements_coord['passworTextField'][1][0]  ]

			if(el_image.shape[0] != 0):
				config = ("-l eng --oem 1 --psm 7")
				text = image_to_string(el_image, config=config)
				print(text)

		
		
	cv2.rectangle(image, elements_coord[eid][0], elements_coord[eid][1], color, 3)


	resized = imutils.resize(image, width = int(image_gray.shape[1] * 0.4))
	cv2.imshow("Image", resized)
	if cv2.waitKey(25) & 0xFF == ord('q'):
		break           

cap.release()
cv2.destroyAllWindows()



