#ayse betul ebren 
#05.12.2021
#camera trajectory estimation assignment 

import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys

#camera matrix with intrinsic parameters
camera_matrix = np.array(
                         [[100, 0, 960],
                         [0, 100, 540],
                         [0, 0, 1]], dtype = "double"
                         )
#distortion coefficients 
dist_coeffs = np.zeros((4,1))

#image point arrays to be detected by ORB
image_points0= np.zeros(shape=(20,2))
image_points1= np.zeros(shape=(20,2))
image_points2= np.zeros(shape=(20,2))

#read 2d-3d arrays
image_points=np.load('vr2d.npy')
model_points=np.load('vr3d.npy')

#read images
img0 = cv2.imread("img1.png");
img1 = cv2.imread("img2.png");
img2 = cv2.imread("img3.png");

#detect orb features
orb0 = cv2.ORB_create(nfeatures=1500)
keypoints_orb0, descriptors0 = orb0.detectAndCompute(img0, None)
orb1 = cv2.ORB_create(nfeatures=1500)
keypoints_orb1, descriptors1 = orb1.detectAndCompute(img1, None)
orb2 = cv2.ORB_create(nfeatures=1500)
keypoints_orb2, descriptors2 = orb2.detectAndCompute(img2, None)

#visualize detected features
cv2.drawKeypoints(img0,keypoints_orb0,img0, color=(255,0,0))
cv2.drawKeypoints(img1,keypoints_orb1,img1, color=(255,0,0))
cv2.drawKeypoints(img2,keypoints_orb2,img2, color=(255,0,0))
imS0 = cv2.resize(img0, (960, 540))  
imS1 = cv2.resize(img1, (960, 540))  
imS2 = cv2.resize(img2, (960, 540))  
#cv2.imshow("firstImageKP",imS0)
#cv2.imshow("secondImageKP",imS1)
#cv2.imshow("thirdImageKP",imS2)
#cv2.waitKey(0)


#take the first 20 keypoints 
for i in range(0,20):
    x, y = keypoints_orb0[i].pt
    image_points0[i]=x,y

for i in range(0,20):
    x, y = keypoints_orb1[i].pt
    image_points1[i]=x,y

for i in range(0,20):
    x, y = keypoints_orb2[i].pt
    image_points2[i]=x,y


#calibrate camera
size=img0.shape
retval,cam_matrix,dist_coefficients,rvecs,tvecs = cv2.calibrateCamera([model_points],[image_points],size[:2],camera_matrix,dist_coeffs,flags=cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_PRINCIPAL_POINT+cv2.CALIB_FIX_ASPECT_RATIO+cv2.CALIB_ZERO_TANGENT_DIST +cv2.CALIB_FIX_K1+cv2.CALIB_FIX_K2+cv2.CALIB_FIX_K3 )


#match descriptors of first and second image
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors0, descriptors1)
matches = sorted(matches, key = lambda x:x.distance)
matching_result = cv2.drawMatches(img0, keypoints_orb0, img1, keypoints_orb1, matches[:20], None, flags=2)

#visualize matching result
imMA = cv2.resize(matching_result, (960, 540))  
#cv2.imshow("MatchRes1",imMA)
#cv2.waitKey(0)

#match descriptors of second and third image
bf1 = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches1 = bf.match(descriptors1, descriptors2)
matches1 = sorted(matches, key = lambda x:x.distance)
matching_result1 = cv2.drawMatches(img1, keypoints_orb1, img2, keypoints_orb2, matches[:20], None, flags=2)

#visualize matching result
imMA1 = cv2.resize(matching_result, (960, 540))  
#cv2.imshow("MatchRes2",imMA1)
#cv2.waitKey(0)

#find rotation matrix and translation vector second image w.r.t first image
E, mask = cv2.findEssentialMat(image_points0, image_points1,cam_matrix,
                               method=cv2.RANSAC, prob=0.999, threshold=3.0 )
retval,R,T,mask=cv2.recoverPose(E, image_points0, image_points1, cam_matrix, mask);

#find rotation matrix and translation vector third image w.r.t first image
E1, mask1 = cv2.findEssentialMat(image_points0, image_points2,cam_matrix,
                               method=cv2.RANSAC, prob=0.999, threshold=3.0 )
retval1,R1,T1,mask1=cv2.recoverPose(E1, image_points0, image_points2, cam_matrix, mask1);

sys.stdout = open("RotationMatrixes.txt", "w")
print('second image rotation matrix w.r.t first image')
print(R)
print('second image translation vector')
print(T)
print('\n')


print('third image rotation matrix w.r.t first image')
print(R1)
print('third image translation vector')
print(T1)
print('\n')
