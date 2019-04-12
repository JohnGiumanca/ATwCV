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
tm_threshold_cursor = 0.6
tm_threshold_elements = 0.7
tm_threshold_page = 100000
intensity_threshold = 5
current_page = None
click = False
animation_in_progress = False

# Elements load
elements = el.load_elements(elements_path, elements_img_type)

cursor = cv2.imread(cursor_path)
cursor = cv2.cvtColor(cursor, cv2.COLOR_BGR2GRAY)
# cursor = cv2.Canny(cursor, 50, 200)

cap = cv2.VideoCapture(input_path)

if cap.isOpened() == False:
	print("Error opening video stream or file!")

_ , first_frame = cap.read()

elements_coord = el.get_elements_coordinates(elements, first_frame, tm_threshold_elements)
elements_color_diff = el.get_elements_color_diff(elements, elements_coord, first_frame)

old_frame = first_frame
while cap.isOpened():
	ret , frame = cap.read()
	if ret == False:
		break
	new_page = False
	keyframe = el.check_keyframe(frame,old_frame, tm_threshold_page)
	if keyframe:
		animation_in_progress = True
		elements_coord = el.get_elements_coordinates(elements, frame, tm_threshold_elements)
		elements_color_diff = el.get_elements_color_diff(elements, elements_coord, frame)
	elif animation_in_progress:
		new_page = True
		animation_in_progress = False

	if new_page:
		print('New Page')
	old_frame = frame

	image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	(startX, startY,endX, endY) = el.find_element(image_gray, cursor, tm_threshold_cursor)

	# draw a bounding box around the detected result 
	view_frame = frame.copy()
	cv2.rectangle(view_frame, (startX, startY), (endX, endY), (51, 51, 255), 3)
	for eid in elements.keys():
		color = (51, 255, 153)

		if elements_coord[eid] != None and el.do_overlap( elements_coord[eid][0], elements_coord[eid][1],
															(startX, startY), (endX, endY)):
			color = (51, 225, 255)
			el_image = view_frame[	elements_coord[eid][0][1]:elements_coord[eid][1][1],
								elements_coord[eid][0][0]:elements_coord[eid][1][0]  ]
			avg1 = cv2.mean(elements[eid])[0:3]
			avg2 = cv2.mean(el_image)[0:3] 
			intensity_diff = abs(elements_color_diff[eid] - el.color_diff(avg1,avg2))
			
			if intensity_diff > intensity_threshold:
				click = True

			if click and (new_page or intensity_diff < intensity_threshold):
				# print(abs(elements_color_diff[eid] - el.color_diff(avg1,avg2)))
				print("Element " + str(eid) + " pressed!")
				click = False

			
			
		cv2.rectangle(view_frame, elements_coord[eid][0], elements_coord[eid][1], color, 3)


	resized = imutils.resize(view_frame, width = int(image_gray.shape[1] * 0.4))
	cv2.imshow("Video", resized)

	if cv2.waitKey(25) & 0xFF == ord('q'):
		break           

cap.release()
cv2.destroyAllWindows()



