import cv2
import random
import numpy as np
from matplotlib import pyplot as plt

img_prime = cv2.imread('practice.jpg')
cv2.imshow('prime', img_prime)
# print(img_prime.shape) 获取原图大小，进行resize

row =320.0/img_prime.shape[1]
dim =(320,int(img_prime.shape[0]*row))
img_resized =cv2.resize(img_prime,dim,interpolation=cv2.INTER_AREA) #用线性插值法进行图像的resize
# print(resized.shape) 输出resize之后的图像大小
cv2.imshow("img_resized",img_resized)

# change color
B, G, R = cv2.split(img_resized)
def random_light_color(img_resized):
    # brightness
    B, G, R = cv2.split(img_resized)

    b_rand = random.randint(-50, 50)
    if b_rand == 0:
        pass
    elif b_rand > 0:
        lim = 255 - b_rand
        B[B > lim] = 255
        B[B <= lim] = (b_rand + B[B <= lim]).astype(img_resized.dtype)
    elif b_rand < 0:
        lim = 0 - b_rand
        B[B < lim] = 0
        B[B >= lim] = (b_rand + B[B >= lim]).astype(img_resized.dtype)

    g_rand = random.randint(-50, 50)
    if g_rand == 0:
        pass
    elif g_rand > 0:
        lim = 255 - g_rand
        G[G > lim] = 255
        G[G <= lim] = (g_rand + G[G <= lim]).astype(img_resized.dtype)
    elif g_rand < 0:
        lim = 0 - g_rand
        G[G < lim] = 0
        G[G >= lim] = (g_rand + G[G >= lim]).astype(img_resized.dtype)

    r_rand = random.randint(-50, 50)
    if r_rand == 0:
        pass
    elif r_rand > 0:
        lim = 255 - r_rand
        R[R > lim] = 255
        R[R <= lim] = (r_rand + R[R <= lim]).astype(img_resized.dtype)
    elif r_rand < 0:
        lim = 0 - r_rand
        R[R < lim] = 0
        R[R >= lim] = (r_rand + R[R >= lim]).astype(img_resized.dtype)

    img_merge = cv2.merge((B, G, R))
    #img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img_merge
img_random_color = random_light_color(img_resized)
cv2.imshow('img_random_color', img_random_color)

img_crop = img_resized[60:320, 30:250] #裁剪图像范围（显示的范围）
cv2.imshow('img_crop', img_crop)

R = cv2.getRotationMatrix2D((img_resized.shape[1] / 2, img_resized.shape[0] / 2), 45, 0.8) #以图像中点为旋转中心，进行图像旋转45度，并且图像缩放为原来的0.8倍
img_rotate = cv2.warpAffine(img_resized, R, (img_resized.shape[1], img_resized.shape[0]))
cv2.imshow('img_rotate', img_rotate)

T = np.float32([[1,0,100],[0,1,100]]) #定义图像平移量
img_transform =cv2.warpAffine(img_resized,T,(img_resized.shape[1],img_resized.shape[0]))
cv2.imshow("img_transform",img_transform)

pts1=np.float32([[56,65],[238,52],[28,237],[239,240]])
pts2 = np.float32([[0,0],[200,0],[0,200],[200,200]])
P =cv2.getPerspectiveTransform(pts1,pts2)
img_perspect=cv2.warpPerspective(img_resized,P,(img_resized.shape[1],img_resized.shape[0]))
cv2.imshow("img_perspect",img_perspect)



# def sift_alignment(image_1: str, image_2: str):
#     im1 = cv2.imread(image_1, cv2.IMREAD_GRAYSCALE)
#     im2 = cv2.imread(image_2, cv2.IMREAD_GRAYSCALE)
#
#     sift = cv2.xfeatures2d.SIFT_create()
#     key_points_1, descriptors_1 = sift.detectAndCompute(im1, None)
#     key_points_2, descriptors_2 = sift.detectAndCompute(im2, None)
#
#     bf_matcher = cv2.BFMatcher()  # brute force matcher
#     # matches = bf_matcher.match(descriptors_1, descriptors_2)  # result is not good
#     matches = bf_matcher.knnMatch(descriptors_1, descriptors_2, k=2)
#
#     # Apply ratio test
#     good_matches = []
#     for m, n in matches:
#         if m.distance < 0.6 * n.distance:  # this parameter affects the result filtering
#             good_matches.append([m])
#
#     match_img = cv2.drawMatchesKnn(im1, key_points_1, im2, key_points_2,
#                                    good_matches, None, flags=2)
#     return len(matches), len(good_matches), match_img
#
#
# matches, good_matches, match_img = sift_alignment('1.png', '2.png')
# cv2.imwrite('match.png', match_img)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()