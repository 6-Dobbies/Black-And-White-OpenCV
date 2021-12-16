import numpy as np
import cv2
import sys

# def laplacian(): 
#     img = cv2.imread('OpenCV/test_image/11.jpg') 
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     _, dst = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)

#     height, width = dst.shape 
#     mask1 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]) 
#     mask2 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]]) 
#     mask3 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]) 
#     laplacian1 = cv2.filter2D(dst, -1, mask1) 
#     laplacian2 = cv2.filter2D(dst, -1, mask2) 
#     laplacian3 = cv2.filter2D(dst, -1, mask3) 
#     laplacian4 = cv2.Laplacian(dst, -1) 
#     gaussian = cv2.GaussianBlur(dst, (5, 5), 0) 
#     LoG = cv2.filter2D(gaussian, -1, mask3) 
#     gaussian1 = cv2.GaussianBlur(dst, (5, 5), 1.6) 
#     gaussian2 = cv2.GaussianBlur(dst, (5, 5), 1) 
#     DoG = np.zeros_like(dst) 
#     for i in range(height):
#         for j in range(width): 
#             DoG[i][j] = float(gaussian1[i][j]) - float(gaussian2[i][j]) 
#     cv2.imshow('chess', gray)
#     cv2.imshow('original', dst)
#     cv2.imshow('laplacian1', laplacian1.astype(float)) 
#     cv2.imshow('laplacian2', laplacian2.astype(float)) 
#     cv2.imshow('laplacian3', laplacian3.astype(float)) 
#     cv2.imshow('laplacian4', laplacian3.astype(float)) 
#     cv2.imshow('LoG', LoG.astype(float))
#     cv2.imshow('DoG', DoG) 
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# laplacian()



#===========================================================

src = cv2.imread('OpenCV/test_image/11.jpg', cv2.IMREAD_COLOR)
dst = cv2.resize(src, dsize=(640, 480), interpolation=cv2.INTER_AREA)

if src is None:
    print('Image load failed!')
    sys.exit()


gray = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)
ret, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
binary = cv2.bitwise_not(binary)

contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

for i in range(len(contours)):
    cv2.drawContours(dst, [contours[i]], 0, (0, 0, 255), 2)
    cv2.putText(dst, str(i), tuple(contours[i][0][0]), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 1)
    print(i, hierarchy[0][i])

alpha = 1 # 기울기
dst2 = np.clip(((1 + alpha) * src - 128 * alpha), 0, 255).astype(np.uint8).copy()

cv2.imshow('src', src)
cv2.imshow('dst', dst2)
cv2.waitKey(0)

cv2.destroyAllWindows()