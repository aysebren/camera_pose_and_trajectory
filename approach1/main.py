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
dist_coeffs = np.zeros((5,1))

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

cv2.imwrite("img1KP.png",imS0);
cv2.imwrite("img1KP.png",imS1);
cv2.imwrite("img1KP.png",imS2);

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



size=img0.shape
#retval,cam_matrix,dist_coefficients,rvecs,tvecs = cv2.calibrateCamera([model_points],[image_points],size[:2],camera_matrix,dist_coeffs,flags=cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_PRINCIPAL_POINT+cv2.CALIB_FIX_ASPECT_RATIO+cv2.CALIB_ZERO_TANGENT_DIST +cv2.CALIB_FIX_K1+cv2.CALIB_FIX_K2+cv2.CALIB_FIX_K3 )

#match descriptors of first and second image
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors0, descriptors1)
matches = sorted(matches, key = lambda x:x.distance)
matching_result = cv2.drawMatches(img0, keypoints_orb0, img1, keypoints_orb1, matches[:20], None, flags=2)

#visualize matching result
imMA = cv2.resize(matching_result, (960, 540))  
#cv2.imshow("MatchRes1",imMA)
#cv2.imwrite("img1img2matchKP.png",imMA);
#cv2.waitKey(0)

#match descriptors of second and third image
bf1 = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches1 = bf1.match(descriptors1, descriptors2)
matches1 = sorted(matches1, key = lambda x:x.distance)
matching_result1 = cv2.drawMatches(img1, keypoints_orb1, img2, keypoints_orb2, matches1[:20], None, flags=2)

#visualize matching result
imMA1 = cv2.resize(matching_result1, (960, 540))  
#cv2.imshow("MatchRes2",imMA1)
#cv2.imwrite("img2img3matchKP.png",imMA1);
#cv2.waitKey(0)

 #solve for rotation and translation vectors for all three frames
(success, rotation_vector0, translation_vector0) = cv2.solvePnP(model_points, image_points0, camera_matrix, dist_coeffs)
(success, rotation_vector1, translation_vector1) = cv2.solvePnP(model_points, image_points1, camera_matrix, dist_coeffs)
(success, rotation_vector2, translation_vector2) = cv2.solvePnP(model_points, image_points2, camera_matrix, dist_coeffs)

sys.stdout = open("RotationMatrixes.txt", "w")
#calculate rotation matrixes
rotation_matrix0 = np.zeros(shape=(3,3))
cv2.Rodrigues(rotation_vector0, rotation_matrix0)
print('first image rotation matrix')
print(rotation_matrix0)
print('first image translation vector')
print(translation_vector0)
print('\n')

rotation_matrix1 = np.zeros(shape=(3,3))
cv2.Rodrigues(rotation_vector1, rotation_matrix1)
print('second image rotation matrix')
print(rotation_matrix1)
print('second image translation vector')
print(translation_vector1)
print('\n')

rotation_matrix2 = np.zeros(shape=(3,3))
cv2.Rodrigues(rotation_vector2, rotation_matrix2)
print('third image rotation matrix')
print(rotation_matrix2)
print('third image translation vector')
print(translation_vector2)
sys.stdout.close()

Rotation = [rotation_matrix0 ,rotation_matrix1 , rotation_matrix2]
Translat = [translation_vector0, translation_vector1, translation_vector2]
Rotation = np.array(Rotation)
Translat = np.array(Translat)

#plot translation vector 
norms = [np.linalg.norm(i) for i in Translat]
plt.plot(range(len(norms)), norms)
plt.show()

#Convert all translations vectors to the coordinate system of the first image
Fullrotation= np.eye(3)
TransInCrdnt = []

for i in range(len(Rotation)):  
    TransInCrdnt.append( Fullrotation@Translat[i].copy() )
    Fullrotation = Fullrotation@np.linalg.inv(Rotation[i].copy())
    
TransInCrdnt = np.squeeze( np.array(TransInCrdnt) )
TransInCrdnt.shape
#plot trajectory
traj = []
summ = np.array([0.,0.,0.])

for i in range(TransInCrdnt.shape[0]):
    traj.append(summ)
    summ = summ + TransInCrdnt[i]
    
traj = np.array(traj)
plt.plot(traj[:,1], traj[:,0])
plt.show()

#Draw the camera trajectory on the first image
plt.figure(figsize=(9,6))
plt.imshow( cv2.imread("img1.png",cv2.IMREAD_UNCHANGED), extent=[600, -600, 600, -600])
plt.plot(-1*traj[:3,0], 1*traj[:3,1],linewidth=4,c='red')

plt.savefig('trajectoryResult.png')
plt.show()