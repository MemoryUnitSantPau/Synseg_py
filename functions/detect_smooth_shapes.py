import numpy as np
import cv2

def detect_smooth_shapes(gray):
        
        gray = gray.astype(np.uint8)
        # Reduce noise for better detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Apply Hough Circle Transform
        # https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=0, maxRadius=0)
        
        detected_shapes = []

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                center = (circle[0], circle[1])
                radius = circle[2]

                perimeter = cv2.arcLength(np.array([center]), True)
                area = cv2.contourArea(np.array([center]))

                circularity = 4 * np.pi * area / (perimeter * perimeter)
                circularity_percentage = int(circularity * 100)
                if abs(1 - (circle[3] / circle[2])) < 0.2:
                    detected_shapes.append(f"Circle ({circularity_percentage}% circular)")
                else:
                    detected_shapes.append(f"Oval ({circularity_percentage}% circular)")
        
        return detected_shapes
