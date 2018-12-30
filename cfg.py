

plate_cfg={
	#ratio thresholds for detecting bounding boxes of chars form plate image
	#for example, the min height of a character in a license plate should be at least 0.2*hight of the plate
	'width_range_threshold': (0.03, 0.3),
	'height_range_threshold': (0.2, 0.7),
	#the height different of each character should be <10%
	'height_diff_threshold': 0.1,
	#if the y_axis different of two characters is >0.75*avg height of characters, then they are assumed to be in different line
	'same_line_threshold': 0.75,
	#padding of image to detect contours that are too close to border(pixel)
	'image_padding': 3,
	#rescalling char roi before ocr, higher value might be useful when detecting blur images.
	'char_resize_factor': 1,
	#padding of the roi before used for tesseract ocr(pixel)
	'char_padding': 15,
	#config string for tesseract
	'config_str': '--psm 10 --oem 1'
}


#===========================================================Plate config/Yolo Config====================================================

yolo_cfg={
	#yolo model weight path for detection
	'model_path': 'model_data/plate_detect.h5',
	#anchors path for yolo
	'anchors_path': 'model_data/anchors.txt',
	#classes path for yolo
	'classes_path': 'model_data/classes.txt',
	#detection threshold for yolo
	'score': 0.5,
	#iou threshold for yolo
	'iou': 0.5,
	#model input image size for yolo
	'model_image_size': (416, 416),
	#gpu_num
	'gpu_num': 1
}
