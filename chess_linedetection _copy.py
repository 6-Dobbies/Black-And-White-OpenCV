import numpy as np
import cv2
import sys
import math
import matplotlib.pyplot as plt


src = cv2.imread('OpenCV/test_image/2D/c7.jfif')

if src is None: 
    print("이미지를 찾을 수 없습니다.")
    sys.exit()

src = cv2.resize(src, dsize=(640, 480), interpolation=cv2.INTER_AREA)
# gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

#자동으로 binary 이미지 전환, 특정값 이상일때 흑과 백으로 표현
gray = cv2.GaussianBlur(gray,(9,9),0)
thresh_value, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

dst = cv2.Canny(gray, 50, 200, None, 3)

line_image = np.copy(src) * 0
lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)

if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv2.line(line_image, pt1, pt2, (255,255,255), 1)


gray_line = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)

lines_edges = cv2.addWeighted(src, 0.8, line_image, 1, 0)

cv2.imshow("line", line_image)
cv2.imshow("res", lines_edges)
cv2.imshow('gray_test', gray)

cv2.imwrite('OpenCV/test_image/2D_gray_test.jpg', gray)

cv2.waitKey()