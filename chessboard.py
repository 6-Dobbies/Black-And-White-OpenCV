import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt

# 캐니엣지 한번 찾아보기, 급격하게 색이 바뀌는걸 강조표시

#이미지 불러오기 경로
src = cv2.imread('OpenCV/test_image/t_image/2d_test.jfif')

#이미지를 찾을수 없을시 콘솔창에 출력되는 문구
if src is None: 
    print("이미지를 찾을 수 없습니다.")
    sys.exit()

#이미지 크기 조절 ( 테스트때 한눈에 보기 편하기 위한 사이즈 조절)

src = cv2.resize(src, dsize=(640, 480), interpolation=cv2.INTER_AREA)

#캐니에지, 대안님 요청= 선은 인식하였으나 흰백 장판 알수없음, 검은바탕에 흰색 선으로만 확인 가능
def canny():
    src = cv2.imread('OpenCV/test_image/t_image/2d_test.jfif')

    edge = cv2.Canny(src, 200, 320)
    cv2.imshow('canny_src', edge)
canny()

#이미지 평탄화(밝기값 위/아래 맞추기)
height = src.shape[0]
width = src.shape[1]

#이미지를 b, g, r 로 나눠서 각각 평탄화
b, g, r = cv2.split(src)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl_b = clahe.apply(b)
cl_g = clahe.apply(g)
cl_r = clahe.apply(r)

src = cv2.merge([cl_b, cl_g, cl_r])

#contrast 기울기 (대비)
alpha = 10.0
dst = np.clip(((1 + alpha) * src - 128 * alpha), 0, 255).astype(np.uint8)

#입력된 변환값 확인을 위한 이미지 창띄우기
cv2.imshow('dst', dst)

src = dst

gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

#해당 부분 주석시 모서리 위주로만 인식
# kernel = np.ones((3, 3), np.uint8)
# gray = cv2.dilate(gray, kernel, iterations=2)
# gray = cv2.erode(gray, kernel, iterations=2)

#자동으로 binary 이미지 전환, 특정값 이상일때 흑과 백으로 표현
gray = cv2.GaussianBlur(gray,(9,9),0)
thresh_value, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

gray = np.float32(gray)
dst = cv2.cornerHarris(gray, 20, 3, 0.04)

# dst = cv2.dilate(dst,None)

#이미지에 따라 임계값이 다름, 확인 필요 
src[dst>0.01*dst.max()]=[0,255,0]

# #최종본 확인 및 저장 종료 적용
cv2.imshow('cornerHarris',src)
cv2.imwrite('OpenCV/test_image/t_image/76.jpg', src)
cv2.waitKey()