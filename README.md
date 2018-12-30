# license-plate-detection-recognition
Detection and recognition for HK car license plate using YOLOv3 (keras) &amp; tesseract &amp; openCV

First of all, the weight of yolo model for plate detection is not open to public. It was trained by 600 images (private dataset). If you are interested please contact me by email.

The task is sperate into two part. Detection of licese plate and recognition of the plate. Please see a simple demo in demo.ipynb

## Credit
YOLOv3 in keras:
https://github.com/qqwweee/keras-yolo3
Tesseract OCR:
https://github.com/tesseract-ocr/tesseract

## Getting Started
If you want to perform dection of license plate, you may have to train it by your own. Following the training instructions in https://github.com/qqwweee/keras-yolo3 , name the trained weight file as plate_detect.h5, the anchors file as anchors.txt. Move both files to model_data directory.
However, if you want to do the detection part using your own model, that's totally fine.

The recogition part is done by using tesseract. So please follow the installation guidline in
https://github.com/tesseract-ocr/tesseract
and
https://pypi.org/project/pytesseract/

## Demo

Please see a simple demo in demo.ipynb
The detection of license plate is done by
```
out_boxes, out_scores, out_classes = detector.dectect_image(image)
```
The recognition of the plate is done by
```
text = recogniser.plate_recognise(plate)
```

## cfg.py
plate_cfg contains the thresholds for filtering of contours, sorting of contours, etc.
The thresholds are used for Hong Kong license plate recognition, it might or might not fit your task. Please change those thresholds for your task.
yolo_cfg cotains the min confidence, iou for the detection job, as well as the gpu assignment.
