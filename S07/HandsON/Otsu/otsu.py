import cv2 
import matplotlib.pyplot as plt 

image = cv2.imread("meter.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7, 7), 0)
(T, thresh) = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY)
(T, threshOtsu) = cv2.threshold(blurred, 0, 255, 
                cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imwrite('a.png', thresh)
cv2.imwrite("au.jpg", threshOtsu)