

plate_cfg={
	#thresholds for detecting bounding boxes of chars form plate image
	'width_range_threshold': (0.03, 0.3),
	'height_range_threshold': (0.2, 0.7),
	'height_diff_threshold': 0.1,
	#threshold of different of centre point for assuming letter are in same line on the plate
	'same_line_threshold': 0.75,
	#padding of image to detect contours that are too close to border(pixel)
	'image_padding': 3,
	#rescalling char roi before ocr
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
