from typing import final
import numpy as np
import cv2 as cv
import os
from matplotlib import pyplot as plt
from scipy.optimize import least_squares

from bokeh.models.widgets import Panel, Tabs

def load_calib(filepath):
    """
    Loads the calibration of the camera
    Parameters
    ----------
    filepath (str): The file path to the camera file
    Returns
    -------
    K_l (ndarray): Intrinsic parameters for left camera. Shape (3,3)
    P_l (ndarray): Projection matrix for left camera. Shape (3,4)
    K_r (ndarray): Intrinsic parameters for right camera. Shape (3,3)
    P_r (ndarray): Projection matrix for right camera. Shape (3,4)
    """
    with open(filepath, 'r') as f:
        params = np.fromstring(f.readline()[4:], dtype=np.float64, sep=' ')
        P_l = np.reshape(params, (3, 4))
        K_l = P_l[0:3, 0:3]
        params = np.fromstring(f.readline()[4:], dtype=np.float64, sep=' ')
        P_r = np.reshape(params, (3, 4))
        K_r = P_r[0:3, 0:3]
    return K_l, P_l, K_r, P_r

def readImages(Path):
    """
    Read Images in specific Path
    
    Parameters:
    Path - Path the contains image_0 and image_1 regarding KITTI dataset

    Returns:
    LeftImgs and RightImgs - Contains the images related to a pair of stereo camera.
    """
    PathLeftImgs = [os.path.join(Path + r"\image_0", file) for file in sorted(os.listdir(Path + r"\image_0"))]
    PathRightImgs = [os.path.join(Path + r"\image_1", file) for file in sorted(os.listdir(Path + r"\image_1"))]
    
    LeftImgs = [cv.imread(path, cv.IMREAD_GRAYSCALE) for path in PathLeftImgs]
    RightImgs  = [cv.imread(path, cv.IMREAD_GRAYSCALE) for path in PathRightImgs]

    return LeftImgs, RightImgs

def calculateDisparityMaps(LeftImgs, RightImgs):
    """
    Calculate Disparity Maps regarding a batch of pair of images
    
    Parameters:
    LeftImgs - Contain a bath of Left Images in a Pair of Images (stereo)
    RightImgs - Contain a bath of Left Images in a Pair of Images (stereo)

    Returns:
    DisparityMaps - Contain the result of Disparity Maps of a batch of pair of images.
    """
    block = 13
    P1 = block * block * 8
    P2 = block * block * 32
    stereo = cv.StereoSGBM_create(minDisparity=0, numDisparities=64, blockSize=block, P1 = P1, P2=P2)
    
    DisparityMaps = []

    for i in range(len(LeftImgs)):    
        DisparityMaps.append(stereo.compute(LeftImgs[i],RightImgs[i]).astype(np.float32)/16)

    return DisparityMaps

def detectKeypoints(Imgs, detector, kp_number, tile_x, tile_y):
    """"
    Detect Keypoints considering crops of dimension (tile_x, tile_y) in a Image. It's
    because considering the whole image can deliver keypoints that are focused in some
    specific region of the image. Using tile through whole image will allow to have
    keypoints in the whole image.

    Parameters:
    Imgs - Images that will be extracted Keypoints.
    detector - detector that will be used for this task.
    kp_number - number of main keypoints detected in each tile.
    tile_x - tile in x (width) direction
    tile_y - tile in y (height) direction

    Returns:
    keypoints - Batch of Keypoints that were detected for each image
    """

    keypoints = []

    for i in range(len(Imgs)):
        kp_image = []
        for y in range(0,Imgs[i].shape[0], tile_y):
            for x in range(0, Imgs[i].shape[1], tile_x):
                croppedImg = Imgs[i][y:(y+tile_y),x:(x+tile_x)]
                kp = detector.detect(croppedImg)

                #Correct coordinates
                for pt in kp:
                    pt.pt = (pt.pt[0] + x, pt.pt[1] + y)

                if(len(kp) > kp_number):
                    kp = sorted(kp, key=lambda x:-x.response)
                    kp = kp[:kp_number]

                kp_image.append(kp)
        keypoints.append(np.concatenate(kp_image))

    return keypoints

def track_keypoints(img1, img2, kp1, lk_params):
    mask = np.zeros_like(img1)
    pts1 = np.zeros((len(kp1),1,2), dtype=np.float32)
    for i,kpt in enumerate(kp1):
        pts1[i,:,0] = kpt.pt[0]
        pts1[i,:,1] = kpt.pt[1]
    pts2, st, err = cv.calcOpticalFlowPyrLK(img1, img2, pts1, None, **lk_params)
    mask = np.zeros(st.shape)
    for i in range(len(pts2)):
        if ((st[i] == 1)):
            mask[i] = 1
    good_new = pts2[mask==1]
    good_old = pts1[mask==1]

    height, width = img1.shape
    in_image = np.where(np.logical_and(good_new[:, 1] < height, good_new[:, 0] < width), True, False)
    good_new = good_new[in_image]
    good_old = good_old[in_image]

    return good_old, good_new

def calculate_right_kps(kp1, kp2, disp1, disp2, min_disp=0.0, max_disp=100.0):
    """
    Calculates the right keypoints (feature points)
    Parameters
    ----------
    q1 (ndarray): Feature points in i-1'th left image. In shape (n_points, 2)
    q2 (ndarray): Feature points in i'th left image. In shape (n_points, 2)
    disp1 (ndarray): Disparity i-1'th image per. Shape (height, width)
    disp2 (ndarray): Disparity i'th image per. Shape (height, width)
    min_disp (float): The minimum disparity
    max_disp (float): The maximum disparity
    Returns
    -------
    kp1_l (ndarray): Feature points in i-1'th left image. In shape (n_in_bounds, 2)
    q1_r (ndarray): Feature points in i-1'th right image. In shape (n_in_bounds, 2)
    q2_l (ndarray): Feature points in i'th left image. In shape (n_in_bounds, 2)
    q2_r (ndarray): Feature points in i'th right image. In shape (n_in_bounds, 2)
    """
    def get_idxs(kp, disp):
        kp_idx = kp.astype(int)
        disp = disp.T[kp_idx[:, 0], kp_idx[:, 1]]
        return disp, np.where(np.logical_and(min_disp < disp, disp < max_disp), True, False)
    
    # Get the disparity's for the feature points and mask for min_disp & max_disp
    disp1, mask1 = get_idxs(kp1, disp1)
    disp2, mask2 = get_idxs(kp2, disp2)
    
    # Combine the masks 
    in_bounds = np.logical_and(mask1, mask2)
    
    # Get the feature points and disparity's there was in bounds
    kp1_l, kp2_l, disp1, disp2 = kp1[in_bounds], kp2[in_bounds], disp1[in_bounds], disp2[in_bounds]
    
    # Calculate the right feature points 
    kp1_r, kp2_r = np.copy(kp1_l), np.copy(kp2_l)
    kp1_r[:, 0] -= disp1
    kp2_r[:, 0] -= disp2
    
    return kp1_l, kp1_r, kp2_l, kp2_r

def triangulate_kps(P_l, P_r, kp_l, kp_r):
    """
    Identify in 3D the points

    Parameters:
    P_l, P_r - Camera Matrix for Left and Right Cameras.
    kp_l, kp_r - Keypoints for Left/Right Cameras
    
    Returns:

    """
    # Triangulate points from i-1'th image
    position = cv.triangulatePoints(P_l, P_r, kp_l.T, kp_r.T)
    # Un-homogenize
    position = np.transpose(position[:3] / position[3])
    return position

def reprojection_residuals(dof, q1, q2, Q1, Q2, P_l):
    """
    Calculate the residuals
    Parameters
    ----------
    dof (ndarray): Transformation between the two frames. First 3 elements are the rotation vector and the last 3 is the translation. Shape (6)
    q1 (ndarray): Feature points in i-1'th image. Shape (n_points, 2)
    q2 (ndarray): Feature points in i'th image. Shape (n_points, 2)
    Q1 (ndarray): 3D points seen from the i-1'th image. Shape (n_points, 3)
    Q2 (ndarray): 3D points seen from the i'th image. Shape (n_points, 3)
    Returns
    -------
    residuals (ndarray): The residuals. In shape (2 * n_points * 2)
    """
    # Get the rotation vector
    r = dof[:3]
    # Create the rotation matrix from the rotation vector
    R, _ = cv.Rodrigues(r)
    # Get the translation vector
    t = dof[3:]
    # Create the transformation matrix from the rotation matrix and translation vector
    transf = np.eye(4, dtype=np.float64)
    transf[:3, :3] = R
    transf[:3, 3] = t
    # Create the projection matrix for the i-1'th image and i'th image
    f_projection = np.matmul(P_l, transf)
    b_projection = np.matmul(P_l, np.linalg.inv(transf))
    # Make the 3D points homogenize
    ones = np.ones((q1.shape[0], 1))
    Q1 = np.hstack([Q1, ones])
    Q2 = np.hstack([Q2, ones])
    # Project 3D points from i'th image to i-1'th image
    q1_pred = Q2.dot(f_projection.T)
    # Un-homogenize
    q1_pred = q1_pred[:, :2].T / q1_pred[:, 2]
    # Project 3D points from i-1'th image to i'th image
    q2_pred = Q1.dot(b_projection.T)
    # Un-homogenize
    q2_pred = q2_pred[:, :2].T / q2_pred[:, 2]
    # Calculate the residuals
    residuals = np.vstack([q1_pred - q1.T, q2_pred - q2.T]).flatten()
    return residuals


def optimize_pose(kp1, kp2, pos_1, pos_2, P_l, P_r, max_iter=100):

    """
    Estimates the transformation matrix
    Parameters
    ----------
    kp1 (ndarray): Feature points in i-1'th image. Shape (n, 2)
    kp2 (ndarray): Feature points in i'th image. Shape (n, 2)
    pos_1 (ndarray): 3D points seen from the i-1'th image. Shape (n, 3)
    pos_2 (ndarray): 3D points seen from the i'th image. Shape (n, 3)
    max_iter (int): The maximum number of iterations
    Returns
    -------
    transformation_matrix (ndarray): The transformation matrix. Shape (4,4)
    """
    early_termination_threshold = 5

    # Initialize the min_error and early_termination counter
    min_error = float('inf')
    early_termination = 0

    for _ in range(max_iter):
        # Choose 5 random feature points
        sample_idx = np.random.choice(range(kp1.shape[0]), 5)
        sample_kp1, sample_kp2, sample_pos_1, sample_pos_2 = kp1[sample_idx], kp2[sample_idx], pos_1[sample_idx], pos_2[sample_idx]

        # Make the start guess
        in_guess = np.zeros(6)
        # Perform least squares optimization
        opt_res = least_squares(reprojection_residuals, in_guess, method='lm', max_nfev=200,
                                args=(sample_kp1, sample_kp2, sample_pos_1, sample_pos_2, P_l))

        # Calculate residuals
        error = reprojection_residuals(opt_res.x, kp1, kp2, pos_1, pos_2, P_l)
        error = error.reshape((pos_1.shape[0] * 2, 2))
        error = np.sum(np.linalg.norm(error, axis=1))

        # Check if the error is less the the current min error. Save the result if it is
        if error < min_error:
            min_error = error
            out_pose = opt_res.x
            early_termination = 0
        else:
            early_termination += 1
        if early_termination == early_termination_threshold:
            # If we have not fund any better result in early_termination_threshold iterations
            break

    # Get the rotation vector
    r = out_pose[:3]
    # Make the rotation matrix
    R, _ = cv.Rodrigues(r)
    # Get the translation vector
    t = out_pose[3:]
    # Make the transformation matrix
    transformation_matrix = (R, t)
    return transformation_matrix

def main():

    #Load Image and CalibrationParameters
    Path = r"C:\Users\Marcos Batista\Documents\git\ComputerVisionProjects\VisualOdometry_Stereo"
    LeftImgs, RightImgs = readImages(Path)
    _l, P_l, K_r, P_r = load_calib(Path + '\calib.txt')  

    #Define Parameters for Lukas-Kanade Method
    lk_params = dict( winSize  = (20,10),
                  maxLevel = 3,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 50, 0.03))

    #Detect Keypoints and Track them between a sequence of Left Images
    fastdetector = cv.FastFeatureDetector_create()
    keypoints = detectKeypoints(LeftImgs, fastdetector, 5, 20, 10)

    #Track and Triangulate Features in order to identify this point in the world (3D)
    tracked_kps = []
    list_pos = np.array([[0,0,0]])
    finalpos = np.zeros(3)
    

    for i in range(len(keypoints)-1):
        tracked_kp1_l, tracked_kp2_l = track_keypoints(LeftImgs[i], LeftImgs[i+1], keypoints[i], lk_params)

        #Calculate Disparity Maps
        DisparityMaps = calculateDisparityMaps(LeftImgs[i:i+2], RightImgs[i:i+2])

        #Calculate Correspodent Keypoint in the Right Image
        kp1_l, kp1_r, kp2_l, kp2_r = calculate_right_kps(tracked_kp1_l, tracked_kp2_l, DisparityMaps[0], DisparityMaps[1])

        #Triangulate Keypoints
        pos_1 = triangulate_kps(P_l, P_r, kp1_l, kp1_r)
        pos_2 = triangulate_kps(P_l, P_r, kp2_l, kp2_r)

        R, t = optimize_pose(kp1_l, kp2_l, pos_1, pos_2, P_l, P_r, max_iter=100)
        finalpos = np.matmul(finalpos.T, R) + t
        np.concatenate(list_pos, finalpos)

        print("Final Pos: ", finalpos)

        #cv.imshow("img",img)
        #cv.waitKey(0)
    fig = plt.figure()

    plt.plot(-list_pos[:,0], list_pos[:,2], 'o')
    # # Data for three-dimensional scattered points
    # ax.scatter3D(list_pos[:,2], list_pos[:,0], list_pos[:,1], cmap='Greens')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


if __name__ == "__main__":
    main()