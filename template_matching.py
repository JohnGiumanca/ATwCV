import argparse
import imutils
import glob
import cv2
import sys
import elements as el



# Notes: 
# 	* cu Canny se face template matching pe edge-uri, de testat
# 	* in get_elements_coordinates se face multiscale pentru elemente, poate dam scara ca parametru
# 	* need to make serious error handlers for the input file. Ex: the elements names form pages file
# 		needs to be the same as the ones from the elements folder

if len(sys.argv) > 1:
	assets_path = sys.argv[1]
else:
	assets_path = '/Users/johnsmacbook/Desktop/ATwCV/Assets'

output_file_name = 'output_file.txt'
output_file = open(output_file_name,'w') 

input_fields_path = assets_path + '/input_fields/'
cursor_path = assets_path + '/cursor.png'
elements_path = assets_path + '/elements/'
input_path = assets_path + '/app_rec.mp4'
elements_img_type = 'png'
pages_path = assets_path + '/pages.txt'
functions_path = assets_path + '/functions.txt'

color_green = (51, 255, 153)
color_red = (51, 51, 255)
color_yellow = (51, 225, 255)

visualize = True             
tm_threshold_cursor = 0.6
tm_threshold_elements = 0.7
tm_threshold_page = 200000
intensity_threshold = 5
current_page = None
click = False
animation_in_progress = False
event_history = [] 			#event_history list, first elements means nothing			

# Elements load
elements = el.load_elements(elements_path, elements_img_type)
cursor = cv2.imread(cursor_path)
cursor = cv2.cvtColor(cursor, cv2.COLOR_BGR2GRAY)
# cursor = cv2.Canny(cursor, 50, 200)

#Element types 
types = el.get_elements_type(elements)

#Pages load
pages = el.load_pages(pages_path)

#Function load
functions = el.load_functions(functions_path)

#test file 
cap = cv2.VideoCapture(input_path)
if cap.isOpened() == False:
	print("Error opening video stream or file!")

#get first frame for initial element search
_ , first_frame = cap.read()
elements_coord = el.get_elements_coordinates(elements, first_frame, tm_threshold_elements)
elements_color_diff = el.get_elements_color_diff(elements, elements_coord, first_frame)

#get current page
current_page = el.get_current_page(elements_coord, pages)
event_history.append('Starting Page - ' + current_page)

#process video frame by frame
old_frame = first_frame
while cap.isOpened():
	ret , frame = cap.read()
	if ret == False:
		break
	
	#check for new page
	new_page = False
	keyframe = el.check_keyframe(frame, old_frame, tm_threshold_page)
	if keyframe:
		animation_in_progress = True
	elif animation_in_progress:
		new_page = True
		animation_in_progress = False
		elements_coord = el.get_elements_coordinates(elements, frame, tm_threshold_elements)
		elements_color_diff = el.get_elements_color_diff(elements, elements_coord, frame)

	if new_page:
		new_current_page = el.get_current_page(elements_coord, pages)
		if new_current_page is not None:
			current_page = new_current_page
		event = 'New Page - ' + current_page
		if event != event_history[-1]:
			event_history.append(event)
			print(event)
	old_frame = frame

	#find cursor
	image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	(startX, startY,endX, endY) = el.find_element(image_gray, cursor, tm_threshold_cursor)

	# draw a bounding box around the cursor
	view_frame = frame.copy()
	cv2.rectangle(view_frame, (startX, startY), (endX, endY), color_red, 3)
	for eid in elements.keys():
		color = color_green

		if elements_coord[eid] != None and el.do_overlap( elements_coord[eid][0], elements_coord[eid][1],
															(startX, startY), (endX, endY)):
			color = color_yellow
			el_image = view_frame[	elements_coord[eid][0][1]:elements_coord[eid][1][1],
								elements_coord[eid][0][0]:elements_coord[eid][1][0]  ]
			avg1 = cv2.mean(elements[eid])[0:3]
			avg2 = cv2.mean(el_image)[0:3] 
			intensity_diff = abs(elements_color_diff[eid] - el.color_diff(avg1,avg2))
			
			if intensity_diff > intensity_threshold:
				click = True

			if click and (animation_in_progress or intensity_diff < intensity_threshold):
				# print(abs(elements_color_diff[eid] - el.color_diff(avg1,avg2)))
				# event = "Element " + str(eid) + " pressed!"
				key = eid,current_page
				event = el.get_event(frame, elements_coord, key, functions,types, input_fields_path)
				if event is not None and event != event_history[-1]:
					event_history.append(event)
					print(event)
				
				click = False


		cv2.rectangle(view_frame, elements_coord[eid][0], elements_coord[eid][1], color, 3)


	if visualize:
		frame = view_frame
	resized = imutils.resize(frame, width = int(image_gray.shape[1] * 0.4))
	cv2.imshow("Video", resized)

	if cv2.waitKey(25) & 0xFF == ord('q'):
		break           

for event in event_history:
	output_file.write(event + '\n')

output_file.close()
cap.release()
cv2.destroyAllWindows()





