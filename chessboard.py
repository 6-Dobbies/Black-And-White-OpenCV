import numpy as np
import cv2
import sys
import chess
import chessboard
import glob

# 경로지정 이미지 불러오기 
# img = cv2.imread('OpenCV/test_image/1.jpg')

# 사각형을 그리는 함수
# def setLabel(img, pts, label):
#     # 사각형 좌표 받아오기
#     (x, y, w, h) = cv2.boundingRect(pts)
#     pt1 = (x, y)
#     pt2 = (x + w, y + h)
#     cv2.rectangle(img, pt1, pt2, (0, 0, 255), 1)
#     cv2.putText(img, label, pt1, cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))


img = cv2.imread('OpenCV/test_image/11.jpg', cv2.IMREAD_COLOR)

# 이미지 사이즈 조절 
dst = cv2.resize(img, dsize=(640, 480), interpolation=cv2.INTER_AREA)
# dst2 = cv2.resize(img, dsize=(0, 0), fx=0.3, fy=0.7, interpolation=cv2.INTER_LINEAR)

gray = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)
ret, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
binary = cv2.bitwise_not(binary)
# binary = cv2.threshold(127,255, cv2.THRESH_BINARY)
# binary = cv2.bitwise_not(binary)

contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

for i in range(len(contours)):
    cv2.drawContours(dst, [contours[i]], 0, (0, 0, 255), 2)
    cv2.putText(dst, str(i), tuple(contours[i][0][0]), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 1)
    print(i, hierarchy[0][i])
    # cv2.imshow("img", img)
    # cv2.waitKey(0)


    
# if img is None:
#     print('Image load failed!')
#     return

# 이미지 경로를 찾을 수 없을때 터미널에 출력
if img is None:
    print("이미지를 찾을 수 없습니다.")
    sys.exit()

# cv2.namedWindow('image')
# # 불러온이미지 팝업창 이름 지정
cv2.imshow("popshow1", dst)
# cv2.imshow("popshow2", dst)
# cv2.imshow("popshow3", dst2)
# # 키보드 입력 전 까지 창을 유지
cv2.waitKey(0)
# 팝업 된 모든 창을 닫음
cv2.destroyAllWindows()

#============================2번쨰 명암 조절============================================#


# src = cv2.imread('OpenCV/test_image/11.jpg', cv2.IMREAD_GRAYSCALE)

# if src is None:
#     print('Image load failed!')
#     sys.exit()

# alpha = 1 # 기울기
# dst = np.clip(((1 + alpha) * src - 128 * alpha), 0, 255).astype(np.uint8).copy()

# cv2.imshow('src', src)
# cv2.imshow('dst', dst)
# cv2.waitKey(0)

# cv2.destroyAllWindows()


# dst = cv2.normalize(src, None, 0, 255, cv2.NORM_MINMAX) # 히스토그램 스트레칭은 NORM_MINMAX

# # 넘파이로 히스토그램 스트레칭 구현
# gmin = np.min(src)
# gmax = np.max(src)
# dst = np.clip(((src - gmin) * 255. / (gmax - gmin), 0, 255).astype(np.unit8))

# cv2.imshow('src', src)
# cv2.imshow('dst', dst)
# cv2.waitKey()

# cv2.destroyAllWindows()