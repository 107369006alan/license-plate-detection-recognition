#detect & recognise the car plate from an image
from yolo import YOLO
from recogniser import Plate_Recogniser
from PIL import Image
import cv2
import sys
import argparse
import numpy as np

def parse_args(args):
    parser = argparse.ArgumentParser(description='Simple HK license plate recognition script')
   
    parser.add_argument('--image', 			  help='Image detection mode, will ignore all positional arguments', default=False, action="store_true")
    parser.add_argument('--video',			  help='Path to the video file which you want to predict')

    return parser.parse_args(args)

#return arr of image from box
def get_box_roi(img, box):
	img = np.asarray(img)
	img = img[box[0]:box[2], box[1]:box[3]]
	return img

def box_in_image(img, box):
	img = np.asarray(img)
	min_height, min_width = 0, 0
	max_height = img.shape[0]
	max_width = img.shape[1]
	if box[0] < 0 or box[1] < 0 or box[2] > max_height or box[3] > max_width:
		return False
	return True

#detect the plate from image and recognise its letters
def image_to_plate_num(detecter, recogniser, img):
	#check if it's square, since the current model is trained by all square dataset
	img = np.asarray(img)
	if img.shape[0] != img.shape[1]:
		print('Not providing square image, using the top part of the image instead')
		img = img[:img.shape[1]]

	img = Image.fromarray(img)
	out_boxes, out_scores, _ = detecter.detect_image(img)

	if len(out_scores) == 0:
		print('No plate detected')
		print(out_boxes, out_scores)
		return [-1], '', -1, -1

	best_box = out_boxes[np.argmax(out_scores)].astype('int').tolist()
	best_score = np.max(out_scores)

	if not box_in_image(img, best_box):
		print('Out of camera')
		return [-1], '', -1, -1

	elif len(out_scores) > 1:
		print('More than one plate detected, picked the plate with highest score for recognition')

	elif len(out_scores) == 1:
		print('One plate detected')

	plate = get_box_roi(img, best_box)
	text = recogniser.plate_recognise(plate, rotate=False)

	print('Box: {}, Text: {}, Score: {}'.format(best_box, text, best_score))
	return best_box, text

def frame_rotate(frame):
    width = frame.shape[0]
    M = cv2.getRotationMatrix2D((width/2,width/2), -90, 1)
    frame = cv2.warpAffine(frame, M, (width, width)) 
    frame = frame[0:width, 0:width]
    return frame
    
#video version of detection and recognition
def video_to_plate_num(detecter, recogniser, video_path):

	boxes = []
	texts = []
	cap = cv2.VideoCapture(video_path)
	video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	frame_num = 0
	while(cap.isOpened()):
		frame_num += 1
		text = ''
		ret, frame = cap.read()
		if not ret:
			break
		if cv2.waitKey(25) & 0xFF == ord('q'):
			break
		#frame = frame_rotate(frame)
		output = image_to_plate_num(detecter, recogniser, frame)
		boxes.append(output[0])
		texts.append(output[1])

		print('========================={}/{}========================='.format(frame_num, video_length))

	return boxes, texts

def main(args=None):

	if args is None:
		args = sys.argv[1:]
	args = parse_args(args)

	detecter = YOLO()
	recogniser = Plate_Recogniser()

	if (args.image and args.video) or (not args.image and not args.video):
		print('Choose one and only one mode from image and video')
		return

	if args.image:
		while True:
			path = input('Please input the image path, input "q" to quit: ')
			start_time = time.time()
			if path == 'q':
				break
			try:
				img = Image.open(path)
			except IOError:
				print ("Error: File does not appear to exist, try again")
				continue
			box, text= image_to_plate_num(detecter, recogniser, img)
			print ('Plate Recognised: {}\nTime Used:{}'.format(text,time.time()-start_time))
		return

	elif args.video:
		boxes, texts = video_to_plate_num(detecter, recogniser, args.video)

		#for demo only
		with open('./demo/video_output/output.txt', 'w') as f:
			print('writing to output file')
			for box, text in zip(boxes,texts):
				f.write('{}|{}\n'.format(','.join(str(coor) for coor in box), text))
main()





