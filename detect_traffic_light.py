import os
import cv2
import argparse
import logging
from math import ceil


# This class provides utility functions to detect the traffic light
class Traffic_Light:
    def __init__(self):
        self.input_path = args.input_path
        self.output_path = args.output_path
        # Assuming the traffic light bbox is static
        self.tl_bbox = [292, 61, 301, 91]
        self.img = None
        self.detected_tf_light = None
        self.detected_color = None

    def detect_contours(self, traffic_light_crop):
        """ Finds contours and returns a bounding box of
            largest contour detected
        """
        gray = cv2.cvtColor(traffic_light_crop, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 150, 255, 0)
        contours, hierarchy = cv2.findContours(thresh,
                                               cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_NONE)
        large_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
        contour_approx = cv2.approxPolyDP(large_contour, 3, True)
        contour_bbox = cv2.boundingRect(contour_approx)

        return contour_bbox

    def draw_detections(self, contour_bbox):
        """ Draws the contour bounding box in the input image
        """
        if self.detected_tf_light:
            detected_bbox = [self.tl_bbox[i] + contour_bbox[i]
                             for i in range(0, len(contour_bbox)-2)]
            detected_bbox.append(detected_bbox[0] + contour_bbox[2])
            detected_bbox.append(detected_bbox[1] + contour_bbox[3])
            cv2.rectangle(self.img, (detected_bbox[0], detected_bbox[1]),
                          (detected_bbox[2], detected_bbox[3]),
                          self.detected_color, 2)
            cv2.putText(self.img, self.detected_tf_light,
                        (self.tl_bbox[0]-5, self.tl_bbox[1]-5),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, self.detected_color,
                        1, cv2.LINE_AA)
        else:
            self.detected_tf_light = "Unknown"
            cv2.putText(self.img, self.detected_tf_light,
                        (self.tl_bbox[0]-5, self.tl_bbox[1]-5),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, self.detected_color,
                        1, cv2.LINE_AA)

    def detect_traffic_light(self):
        ''' Reads images from input path, detects traffic lights and 
            saves the result images
        '''
        i = 0
        contour_window = 2
        # Create output directory
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        # Read images from input path
        for file in os.listdir(self.input_path):
            logging.info("Processing: ", os.path.join(self.input_path, file))

            self.img = cv2.imread(os.path.join(self.input_path, file))
            # Crop the portion of traffic light
            tl_crop = self.img[self.tl_bbox[1]:self.tl_bbox[3],
                               self.tl_bbox[0]:self.tl_bbox[2]]
            bbox_h = self.tl_bbox[3] - self.tl_bbox[1]
            # Find a contour in the crop
            contour_bbox = self.detect_contours(tl_crop)
            # Get the percentage to equi-split the crop region
            tl_split = ceil(0.33 * bbox_h)
            # Condition to check if the detected contour falls in
            # top/middle/bottom portion of the traffic light crop
            if contour_bbox[1] <= tl_split and \
               contour_bbox[1] + contour_bbox[3] <= tl_split + contour_window:
                # Detected bbox falls in the top portion of the crop
                self.detected_tf_light = "Red"
                self.detected_color = (0, 0, 255)
            elif contour_bbox[1] > tl_split - contour_window and \
                    contour_bbox[1] + contour_bbox[3] <= \
                    (bbox_h-tl_split) + contour_window:
                # Detected bbox falls in the middle portion of the crop
                self.detected_tf_light = "Amber"
                self.detected_color = (0, 126, 255)
            elif contour_bbox[1] > (bbox_h - tl_split) - contour_window and \
                    contour_bbox[1] + contour_bbox[3] <= bbox_h:
                # Detected bbox falls in the bottom portion of the crop
                self.detected_tf_light = "Green"
                self.detected_color = (0, 255, 0)
            else:
                self.detected_tf_light = None
                self.detected_color = (255, 0, 0)

            self.draw_detections(contour_bbox)

            i += 1
            cv2.imwrite(os.path.join(self.output_path, file), self.img)

            logging.info("Detected : ", self.detected_tf_light)

        return


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Detect traffic light')
    parser.add_argument('-ip', '--input_path', action='store_true',
                        default='F:\data',
                        help='Enter the directory path having input images')
    parser.add_argument('-op', '--output_path', action='store_true',
                        default='results',
                        help='Enter the directory path to save output images')
    args = parser.parse_args()

    tl = Traffic_Light()
    tl.detect_traffic_light()
