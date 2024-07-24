import numpy as np
import cv2

def detect_shapes(thresh):
    thresh = thresh.astype(np.uint8)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_shapes = []
    
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
        num_sides = len(approx)
        
        shape_name = "Unknown"
        if num_sides == 3:
            shape_name = "Triangle"
        elif num_sides == 4:
            shape_name = "Rectangle" if cv2.contourArea(contour) > 50 else "Square"
        elif num_sides == 5:
            shape_name = "Pentagon"
        else:
            shape_name = "Circle"
        
        detected_shapes.append(shape_name)
    
    return detected_shapes

