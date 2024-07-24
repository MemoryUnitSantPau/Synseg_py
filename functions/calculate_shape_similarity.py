import numpy as np
import cv2
import pandas as pd
def calculate_shape_similarity(img1):
        """
        Based on code from 
        https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_contours/py_contours_more_functions/py_contours_more_functions.html
        """

        shapes = [
            "Circle",
            "Oval",
            "Square"
        ]
        results = pd.DataFrame()

        for shape in shapes:
            
            if shape == "Circle":
                img2 = np.zeros((60, 60), dtype=np.uint8)
                center_x, center_y = 30, 30 
                radius = 10
                y, x = np.ogrid[-radius:radius, -radius:radius]
                circle = x**2 + y**2 <= radius**2
                img2[center_x - radius:center_x + radius, center_y - radius:center_y + radius][circle] = 255
                
            elif shape == "Elipse":
                img2 = np.zeros((60, 60), dtype=np.uint8)
                center_x, center_y = 30, 30  
                semi_major_axis = 20
                semi_minor_axis = 10
                y, x = np.ogrid[-semi_minor_axis:semi_minor_axis, -semi_major_axis:semi_major_axis]
                oval = (x**2) / (semi_major_axis**2) + (y**2) / (semi_minor_axis**2) <= 1
                img2[center_x - semi_major_axis:center_x + semi_major_axis,
                    center_y - semi_minor_axis:center_y + semi_minor_axis][oval] = 255

            elif shape == "Square":
                image = np.zeros((60, 60), dtype=np.uint8)
                center_x, center_y = 30, 30
                side_length = 20
                half_side = side_length // 2
                start_x, end_x = center_x - half_side, center_x + half_side
                start_y, end_y = center_y - half_side, center_y + half_side
                image[start_x:end_x, start_y:end_y] = 255
            
            img1 = img1.astype(np.uint8)
            img2 = img2.astype(np.uint8)
            ret, thresh = cv2.threshold(img1, 127, 255, 0)
            ret, thresh2 = cv2.threshold(img2, 127, 255, 0)
            contours, _ = cv2.findContours(thresh, 2, 1)
            cnt1 = contours[0] if contours else None
            contours, _ = cv2.findContours(thresh2, 2, 1)
            cnt2 = contours[0] if contours else None
            ret = cv2.matchShapes(cnt1, cnt2, 1, 0.0)
            results[shape] = ret
        
        return results
