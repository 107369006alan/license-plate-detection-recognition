
#Recognise a plate!

import pytesseract
from pytesseract import Output
import cv2
import os
import numpy as np
import statistics
import sys
import argparse
#if you need to change the default excutable tesseract
#pytesseract.pytesseract.tesseract_cmd = r'/usr/local/Cellar/tesseract/4.0.0/bin/tesseract'
import time
#self-defined config
import cfg

class Plate_Recogniser(object):

    _defaults = {
        "config_str": cfg.plate_cfg['config_str'],
        "width_range_threshold": cfg.plate_cfg['width_range_threshold'],
        "height_range_threshold": cfg.plate_cfg['height_range_threshold'],
        "height_diff_threshold": cfg.plate_cfg['height_diff_threshold'],
        "same_line_threshold": cfg.plate_cfg['same_line_threshold'],
        "char_padding": cfg.plate_cfg['char_padding'],
        "char_resize_factor": cfg.plate_cfg['char_resize_factor'],
        "image_padding": cfg.plate_cfg['image_padding']
    }

    @classmethod

    def get_defaults(cls, atr):
        if atr in cls._defaults:
            return cls._defaults[atr]
        else:
            return "Unrecognized attribute name '" + atr + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        

    def plate_recognise(self, img, rotate = False):

        #read image as np array
        img = np.asarray(img)


        #get rectangle images of the chars and correspoding centres
        rois, rois_centres = self.get_rois(img = img,
                                            width_range_threshold = self.width_range_threshold,
                                            height_range_threshold = self.height_range_threshold,
                                            height_diff_threshold = self.height_diff_threshold)
    
        if not rois:
            return ''
        
        #get the text and corresponding text centres
        text, text_centres = self.rois_to_text(rois = rois, rois_centres = rois_centres, config = self.config_str)

        if not text:
            return ''

        #get the average height of rois
        def avg_height(rois):
            avg_h = 0
            for roi in rois:
                avg_h += roi.shape[0]
            avg_h /= len(rois)
            return avg_h

        #get the threshold and sort it from up to down left to right
        text = self.sort_chars(text = text, text_centres = text_centres, same_line_threshold = self.same_line_threshold*avg_height(rois))

        return text


    #fliter redundant contours by size, their hierarchy and their height different
    #only need to return rois and centres in real implementation
    def get_rois(self, img, width_range_threshold, height_range_threshold, height_diff_threshold):

        def centre_of_cnt(cnt):
            M = cv2.moments(cnt)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            return [cX, cY]

        #get black-white image
        img_bw = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        img_bw = cv2.threshold(img_bw, 170, 255, cv2.THRESH_OTSU)[1]

        #padding to detect chars that are too close to border
        img_bw = cv2.copyMakeBorder(img_bw, self.image_padding, self.image_padding, self.image_padding, self.image_padding, cv2.BORDER_CONSTANT, value=(255))
        #padding the same value to the ori image to get correct roi
        img = cv2.copyMakeBorder(img, self.image_padding, self.image_padding, self.image_padding, self.image_padding, cv2.BORDER_CONSTANT, value=(255, 255, 255))

        #find all contours in the image
        _, contours, hierarchy = cv2.findContours(img_bw,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None, None

        #get image height, width and use it for getting thresholds
        height = img.shape[0]
        width = img.shape[1]
        min_width = int(width*width_range_threshold[0])
        min_height = int(height*height_range_threshold[0])
        max_width = int(width*width_range_threshold[1])
        max_height = int(height*height_range_threshold[1])
        

        #list of index of reasonable contours
        flitered_contours_index = []

        #region of interest
        rois = []
        #centre coordinates of the contours in original image. Will be used for sorting and rotatation
        rois_centres = []
        #height of rois will be used for filtering
        rois_heights = []

        #first round: fliter them by width and height thresholding
        for i, cnt in enumerate(contours):
            x,y,w,h = cv2.boundingRect(cnt)
            if w < max_width and h < max_height and h > min_height and w > min_width:
                flitered_contours_index.append(i)
            
        if len(flitered_contours_index) == 0:
            return None, None

        #second round: fliter them by their hierarchies
        for i in flitered_contours_index:
            if hierarchy[0][i][3] in flitered_contours_index:
                flitered_contours_index.remove(i)

        if len(flitered_contours_index) == 0:
            return None, None

        #get the height threshold(medium of heights) for third round filtering
        for i in flitered_contours_index:
            _,_,_,h = cv2.boundingRect(contours[i])
            rois_heights.append(h)
        median_height = statistics.median(rois_heights)
        height_diff_threshold = median_height*height_diff_threshold
        
        #last round filtering and get the rois, centres
        for i in flitered_contours_index:
            x,y,w,h = cv2.boundingRect(contours[i])
            #ignore this one if its height is too different from the median height
            if abs(h-median_height) > height_diff_threshold:
                continue
            #get the rois from origional image
            ymin, ymax, xmin, xmax = y, y+h, x, x+w
            roi = img[ymin:ymax, xmin:xmax]
            rois.append(roi)
            rois_centres.append(centre_of_cnt(contours[i]))

        return rois, rois_centres

    #using rois to get plate string!
    def rois_to_text(self, rois, rois_centres, config='--psm 10 --oem 1', handmade_correct = True):

        def roi_preprocessing(roi):
            #rescalling -> binarisation -> padding
            h, w = roi.shape[:2]
            roi = cv2.resize(roi, (w*self.char_resize_factor, h*self.char_resize_factor))
            roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            roi = cv2.threshold(roi, 170, 255, cv2.THRESH_OTSU)[1]
            padding = self.char_padding
            roi = cv2.copyMakeBorder(roi, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=(255))
            return roi

        text = ''
        text_centres = []

        if not rois:
            return text, text_centres
        
        for i, roi in enumerate(rois):

            roi = roi_preprocessing(roi)

            char = pytesseract.image_to_string(roi, config=config)
            #tesseract sometimes give same char twice. maybe my config is broken
            if len(char) > 1:
                char = char[0]
            #if it is already giving reasonable char, pass and add it into text
            elif (char.isupper() or char.isdigit()) and char!='I' and char!='O' and char!='Q':
                pass
            #guessing, please remove below if tesseract whitelist is useable!
            elif char == '|' or char == 'I' or char == 'l' or char=='\\' \
                or char=='/' or char==']':
                char = '1'
            elif char == 'O' or char =='o' or char == 'Q':
                char = '0'
            elif char =='c':
                char == 'C'
            elif char =='k':
                char == 'K'
            elif char == 'm':
                char = 'M'
            elif char == 'p':
                char = 'P'
            elif char == 's':
                char = 'S'
            elif char == 'u':
                char = 'U'
            elif char == 'v':
                char = 'V'
            elif char == 'w':
                char = 'W'
            elif char == 'x':
                char = 'X'
            elif char == 'y':
                char = 'Y'
            elif char == 'z':
                char = 'Z'
            else:
                continue
            #add flitered char into text
            text += char
            text_centres.append(rois_centres[i])
        #TODO: lastly fliter the redundant rois and its corresbonding text and centres by
        #compareing the height of each remaining rois!
        return text, text_centres

    def sort_chars(self, text, text_centres, same_line_threshold):
        if text == '':
            return ''
        all_x = [row[0] for row in text_centres]
        all_y = [row[1] for row in text_centres]
        #zip text, x, y together correspondingly in to one list
        all_in_one = list(zip(text, all_x, all_y))
            
        #if only one line, sort by x axis and done
        if max(all_y) - min(all_y) < same_line_threshold:
            all_in_one = sorted(all_in_one, key=lambda k: k[1])
            return ''.join([row[0] for row in all_in_one])
            
        #two lines case
        #if max_y - y coordinate of the char is larger than threshold, then its grouped into first line
        first_line = [one for one in all_in_one if max(all_y) - one[2] > same_line_threshold]
        #sort the first line by x axis and 
        first_line = sorted(first_line, key=lambda k: k[1])
        sorted_first_line = ''.join([row[0] for row in first_line])
        #all others will be grouped in to second line
        second_line = [one for one in all_in_one if one not in first_line]
        second_line = sorted(second_line, key=lambda k: k[1])
        sorted_second_line = ''.join([row[0] for row in second_line])

        return sorted_first_line+sorted_second_line

