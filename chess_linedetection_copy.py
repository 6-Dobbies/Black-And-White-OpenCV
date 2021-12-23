import numpy as np
import cv2
import sys
import math
import matplotlib.pyplot as plt
from collections import defaultdict

src = cv2.imread('OpenCV/test_image/2D/test.jpg')

if src is None: 
    print("이미지를 찾을 수 없습니다.")
    sys.exit()

src = cv2.resize(src, dsize=(640, 480), interpolation=cv2.INTER_AREA)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)


#자동으로 binary 이미지 전환, 특정값 이상일때 흑과 백으로 표현
gray = cv2.GaussianBlur(gray,(9,9),0)
thresh_value, gray = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

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

lines = np.reshape(lines, (-1, 2))

def h_v_lines(lines):
    h_lines, v_lines = [], []
    for rho, theta in lines:
        if theta < np.pi / 4 or theta > np.pi - np.pi / 4:
            v_lines.append([rho, theta])
        else:
            h_lines.append([rho, theta])
    return h_lines, v_lines

h_lines, v_lines = h_v_lines(lines)

def line_intersections(h_lines, v_lines):
    points = []
    for r_h, t_h in h_lines:
        for r_v, t_v in v_lines:
            a = np.array([[np.cos(t_h), np.sin(t_h)], [np.cos(t_v), np.sin(t_v)]])
            b = np.array([r_h, r_v])
            inter_point = np.linalg.solve(a, b)
            points.append(inter_point)
    return np.array(points)

points = line_intersections(h_lines, v_lines)
points = np.array(sorted(points, key = lambda x:x[0]))
points = np.array(sorted(points, key = lambda x:x[1]))

X = list(set([int(i[0]) for i in points]))
Y = list(set([int(i[1]) for i in points]))

X.sort()
Y.sort()

img_count = 0

for i in range(0, len(X)-1):
    for j in range(0, len(Y)-1):
        cropped = src[Y[j]: Y[j+1], X[i]: X[i+1]]
        img_count += 1
        cv2.imwrite('OpenCV/test_image/2D/test/' + str(img_count) + '.jpeg', cropped)
        print('OpenCV/test_image/2D/test/' + 'data' + str(img_count) + '.jpeg')


lines_edges = cv2.addWeighted(src, 0.8, line_image, 1, 0)

cv2.imshow('line', line_image)
cv2.imshow("res", lines_edges)
cv2.waitKey()