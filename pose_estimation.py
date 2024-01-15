"""
Implements 2 tasks.
1. Given a april tag with known pose relative to a base frame and known size and intrinsic camera parameters, estimate the pose of the camera relative to the tag base frame.
2. Given a known camera pose relative to a base tag, estimate the pose of a target tag relative to the base tag.

Note that cv2.solvePnP() returns the transformation from the object coordinate frame to the camera coordinate frame.
We want the inverse of that transformation so that we can get the position of the camera relative to the object.
"""

import cv2
import numpy as np
from scipy.optimize import least_squares, approx_fprime
import scipy.sparse as sparse

from visualization import initialize_viz, plot_coordinate_frame, plot_corners, set_aspect_ratio

def dcm_from_rpy(rpy):
    """
    Rotation matrix from roll, pitch, yaw Euler angles.

    The function produces a 3x3 orthonormal rotation matrix R
    from the vector rpy containing roll angle r, pitch angle p, and yaw angle
    y.  All angles are specified in radians.  We use the aerospace convention
    here (see descriptions below).  Note that roll, pitch and yaw angles are
    also often denoted by phi, theta, and psi (respectively).

    The angles are applied in the following order:

     1.  Yaw   -> by angle 'y' in the local (body-attached) frame.
     2.  Pitch -> by angle 'p' in the local frame.
     3.  Roll  -> by angle 'r' in the local frame.  

    Note that this is exactly equivalent to the following fixed-axis
    sequence:

     1.  Roll  -> by angle 'r' in the fixed frame.
     2.  Pitch -> by angle 'p' in the fixed frame.
     3.  Yaw   -> by angle 'y' in the fixed frame.

    Parameters:
    -----------
    rpy  - 3x1 np.array of roll, pitch, yaw Euler angles.

    Returns:
    --------
    R  - 3x3 np.array, orthonormal rotation matrix.
    """
    cr = np.cos(rpy[0]).item()
    sr = np.sin(rpy[0]).item()
    cp = np.cos(rpy[1]).item()
    sp = np.sin(rpy[1]).item()
    cy = np.cos(rpy[2]).item()
    sy = np.sin(rpy[2]).item()

    return np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                     [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                     [  -sp,            cp*sr,            cp*cr]])

def rpy_from_dcm(R):
    """
    Roll, pitch, yaw Euler angles from rotation matrix.

    The function computes roll, pitch and yaw angles from the
    rotation matrix R. The pitch angle p is constrained to the range
    (-pi/2, pi/2].  The returned angles are in radians.

    Inputs:
    -------
    R  - 3x3 orthonormal rotation matrix.

    Returns:
    --------
    rpy  - 3x1 np.array of roll, pitch, yaw Euler angles.
    """
    rpy = np.zeros((3, 1))

    # Roll.
    rpy[0] = np.arctan2(R[2, 1], R[2, 2])

    # Pitch.
    sp = -R[2, 0]
    cp = np.sqrt(R[0, 0]*R[0, 0] + R[1, 0]*R[1, 0])

    if np.abs(cp) > 1e-15:
      rpy[1] = np.arctan2(sp, cp)
    else:
      # Gimbal lock...
      rpy[1] = np.pi/2
  
      if sp < 0:
        rpy[1] = -rpy[1]

    # Yaw.
    rpy[2] = np.arctan2(R[1, 0], R[0, 0])

    return rpy

# def get_tag_corners_mm(T_1_Ai, tag_size):
#     """
#     Gets the 3D coordinates of the corners of the tag in the base frame.
#     """
#     # tag_corners_mm_Ai = np.array([
#     #     [-tag_size/2, -tag_size/2, 0],
#     #     [tag_size/2, -tag_size/2, 0],
#     #     [tag_size/2, tag_size/2, 0],
#     #     [-tag_size/2, tag_size/2, 0],
#     #     # [0, 0, 0]
#     # ])
#     tag_corners_mm_Ai = np.array([
#         [-tag_size/2, tag_size/2, 0],
#         [tag_size/2, tag_size/2, 0],
#         [tag_size/2, -tag_size/2, 0],
#         [-tag_size/2, -tag_size/2, 0],
#         # [0, 0, 0]
#     ])
#     # Transform the tag corners from the tag frame to the base frame
#     tag_corners_mm_1 = T_1_Ai @ np.hstack((tag_corners_mm_Ai, np.ones((len(tag_corners_mm_Ai),1)))).T
#     tag_corners_mm_1 = tag_corners_mm_1[:3,:].T
#     return tag_corners_mm_1

def get_tag_corners_mm(T_1_Ai, tag_corners_mm_Ai):
    """
    Gets the 3D coordinates of the corners of the tag in the base frame.
    """
    # Transform the tag corners from the tag frame to the base frame
    tag_corners_mm_1 = T_1_Ai @ np.hstack((tag_corners_mm_Ai, np.ones((len(tag_corners_mm_Ai),1)))).T
    tag_corners_mm_1 = tag_corners_mm_1[:3,:].T
    return tag_corners_mm_1


def estimate_T_Ci_Ai(tag_corners_px, tag_corners_mm_Ai, camera_matrix, dist_coeffs):
    """
    Estimates the transformation from the tag frame to the camera frame.
    tag_corners_px: nx2 numpy array of the corners of the tag in the image in pixel coordinates.
    tag_size: The size of the tag in mm.
    camera_matrix: 3x3 numpy array of the camera intrinsic parameters.
    dist_coeffs: 5x1 numpy array of the camera distortion coefficients.
    Returns:
    T_Ci_Ai: 4x4 numpy array of the transformation from the tag frame to the camera frame.
    """
    # Define the 3D points of the tag corners in the tag frame. We assume that the tag is centered at the origin.
    tag_corners_mm = get_tag_corners_mm(np.eye(4), tag_corners_mm_Ai)

    _, R, t = cv2.solvePnP(tag_corners_mm, tag_corners_px, camera_matrix, dist_coeffs)#, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    
    # Convert the rotation vector to a rotation matrix
    R, _ = cv2.Rodrigues(R)

    # Convert the rotation matrix and translation vector to a transformation matrix
    T_Ci_Ai = np.eye(4)
    T_Ci_Ai[:3,:3] = R
    T_Ci_Ai[:3,3] = t.flatten()

    return T_Ci_Ai

def estimate_camera_pose(tag_corners_px, tag_corners_mm_Ai, camera_matrix, dist_coeffs, T_1_Ai=None):
    """
    Estimates the pose of the camera relative to the base frame.
    tag_corners_px: nx2 numpy array of the corners of the tag in the image in pixel coordinates.
    tag_size: The size of the tag in mm.
    camera_matrix: 3x3 numpy array of the camera intrinsic parameters.
    dist_coeffs: 5x1 numpy array of the camera distortion coefficients.
    T_1_Ai: 4x4 numpy array of the transformation from the tag frame to the base frame. If None, use identity rotation and zero translation.
    Returns:
    T_1_Ci: 4x4 numpy array of the transformation from the camera frame to the base frame.
    """
    if T_1_Ai is None:
        T_1_Ai = np.eye(4)

    # Get the transform from the tag frame to the camera frame
    T_Ci_Ai = estimate_T_Ci_Ai(tag_corners_px, tag_corners_mm_Ai, camera_matrix, dist_coeffs)

    # Invert the transformation matrix to get the transform from the camera frame to the tag frame
    T_Ai_Ci = np.linalg.inv(T_Ci_Ai)

    # Get the transform from the camera frame to the base frame
    T_1_Ci = T_1_Ai @ T_Ai_Ci

    return T_1_Ci

def multi_tag_estimate_camera_pose(tag_poses, tag_corner_px_positions, tag_corners_mm_Ai, camera_matrix, dist_coeffs):
    """
    Estimates the pose of the camera relative to the base frame using multiple tags.
    tag_poses: Dict mapping from tag ID to the pose of the tag in the base frame.
    tag_corner_px_positions: Dict mapping from tag ID to a list of the pixel positions of the tag corners in each image.
    tag_size: The size of the tag in mm.
    camera_matrix: 3x3 numpy array of the camera intrinsic parameters.
    dist_coeffs: 5x1 numpy array of the camera distortion coefficients.
    Returns:
    T_1_Ci: 4x4 numpy array of the transformation from the camera frame to the base frame.
    """
    # First, we need to get the 4nx2 matrix of pixel coordinates of the tag corners and the 4nx3 matrix of 3D coordinates of the tag corners
    # get_tag_corners_mm now returns an unknown number of corners for each tag so we need to do this a little differently
    tag_corners_px_matrix = []
    tag_corners_mm_matrix = []
    for tag_id, tag_pose in tag_poses.items():
        tag_corners_px = tag_corner_px_positions[tag_id]
        tag_corners_mm = get_tag_corners_mm(tag_pose, tag_corners_mm_Ai)
        tag_corners_px_matrix.append(tag_corners_px)
        tag_corners_mm_matrix.append(tag_corners_mm)
    tag_corners_px_matrix = np.vstack(tag_corners_px_matrix)
    tag_corners_mm_matrix = np.vstack(tag_corners_mm_matrix)
    
    # Now we can use cv2.solvePnP() to estimate the camera pose
    _, R, t = cv2.solvePnP(tag_corners_mm_matrix, tag_corners_px_matrix, camera_matrix, dist_coeffs)

    # Convert the rotation vector to a rotation matrix
    R, _ = cv2.Rodrigues(R)

    # Convert the rotation matrix and translation vector to a transformation matrix
    T_Ci_Ai = np.eye(4)
    T_Ci_Ai[:3,:3] = R
    T_Ci_Ai[:3,3] = t.flatten()

    # Invert the transformation matrix to get the transform from the camera frame to the tag frame
    T_1_Ci = np.linalg.inv(T_Ci_Ai)

    # This time we are already in the base frame so we don't need to do anything else
    return T_1_Ci


def estimate_tag_pose(tag_corners_px, tag_corners_mm_Ai, camera_matrix, dist_coeffs, T_1_Ci):
    """
    Estimates the pose of the target tag relative to the base tag.
    tag_corners_px: nx2 numpy array of the corners of the tag in the image in pixel coordinates.
    tag_size: The size of the tag in mm.
    camera_matrix: 3x3 numpy array of the camera intrinsic parameters.
    dist_coeffs: 5x1 numpy array of the camera distortion coefficients.
    T_1_Ci: 4x4 numpy array of the transformation from the camera frame to the base frame.
    Returns:
    T_1_Ai: 4x4 numpy array of the transformation from the target tag frame to the base frame.
    """
    T_Ci_Ai = estimate_T_Ci_Ai(tag_corners_px, tag_corners_mm_Ai, camera_matrix, dist_coeffs)

    # The transformation from the target tag frame to the base frame is the transformation from the camera frame to the base frame times the transformation from the target tag frame to the camera frame
    T_1_Ai = T_1_Ci @ T_Ci_Ai

    return T_1_Ai

def find_corner_px_projection(camera_pose, tag_pose, tag_corners_mm_Ai, camera_matrix, dist_coeffs):
    """
    Transforms the corners of the target tag into the camera frame and then projects them into pixel coordinates.
    """
    # Get the corners of the target tag in the base frame
    tag_corners_mm = get_tag_corners_mm(tag_pose, tag_corners_mm_Ai)

    # Convert to homogeneous coordinates
    tag_corners_mm = np.hstack((tag_corners_mm, np.ones((len(tag_corners_mm),1)))).T
    # Transform to the camera frame
    tag_corners_mm = np.linalg.inv(camera_pose) @ tag_corners_mm
    # Convert back to cartesian coordinates
    tag_corners_mm = tag_corners_mm[:3,:].T

    # Project the corners into pixel coordinates
    tag_corners_px, _ = cv2.projectPoints(tag_corners_mm, np.zeros((3,1)), np.zeros((3,1)), camera_matrix, dist_coeffs)

    # # Or manually project the corners into pixel coordinates
    tag_corners_px_manual = camera_matrix @ tag_corners_mm.T
    tag_corners_px_manual = (tag_corners_px_manual[:2,:] / tag_corners_px_manual[2,:]).T

    return tag_corners_px.reshape(-1, 2)



def find_cam_tag_jacobian(tag_index, camera_index, total_features, total_cameras, tag_pose, camera_pose, tag_corners_mm_Ai, camera_matrix, dist_coeffs, include_tag_jac=True, include_camera_jac=True):
    """
    Computes the jacobian of the error function for bundle adjustment for a single tag and camera pair.
    Stacking many of these jacobians gives the full jacobian for bundle adjustment.

    Each tag has 6 parameters and each camera has 6 parameters.
    Each tag has n corners and each corner has 2 pixel coordinates.
    Therefore, the size of the jacobian is 2nx(6*total_tags + 6*total_cameras). But only 12 of the columns are nonzero.
    """
    # tag_jacobian = np.zeros((8, 6), dtype=np.float64)
    # camera_jacobian = np.zeros((8, 6), dtype=np.float64)
    # expected_pixels = np.zeros((8, 1), dtype=np.float64)
    tag_jacobian = []
    camera_jacobian = []
    expected_pixels = []
    # Pre-baked matrices used to take derivatives of 1 axis rotation matrices
    X_bar_z = np.array([[0, -1, 0],[1, 0, 0],[0, 0, 0]])
    X_bar_y = np.array([[0, 0, 1],[0, 0, 0],[-1, 0, 0]])
    X_bar_x = np.array([[0, 0, 0],[0, 0, -1],[0, 1, 0]])

    # Step 1: Split the poses into rotations and translations
    R_1_At = tag_pose[:3,:3]
    t_1_At = tag_pose[:3,3].reshape(-1, 1)
    R_1_Ci = camera_pose[:3,:3]
    t_1_Ci = camera_pose[:3,3].reshape(-1, 1)

    # And we will be using the inverse of the camera pose to let's just get that now
    R_1_Ci_T = R_1_Ci.T

    # Step 2: Loop through each corner of the tag
    corners = get_tag_corners_mm(np.eye(4), tag_corners_mm_Ai)
    for corner_index in range(len(corners)):
        corner_pos = corners[corner_index,:].reshape(-1, 1)  # (3x1)
        # Step 3: Compute intermediate values
        p_1 = R_1_At @ corner_pos + t_1_At  # Position of the corner in the base frame (3x1)
        P_ci = R_1_Ci_T @ (p_1 - t_1_Ci)  # Position of the corner in the camera frame (3x1)
        y = camera_matrix @ P_ci  # Un-normalized expected pixel coordinates (3x1)
        x_it = y[:2,:] / y[2,:]  # Expected pixel coordinates (2x1)

        # y_2 = camera_matrix @ R_1_Ci_T @ (R_1_At @ corner_pos + t_1_At - t_1_Ci)  # (3x1)
        # x_it_2 = y_2[:2,:] / y_2[2,:]  # Expected pixel coordinates (2x1)

        # # Use the cv2 camera projection function to get the expected pixel coordinates
        x_it_cv2 = cv2.projectPoints(P_ci.T, np.zeros((3,1)), np.zeros((3,1)), camera_matrix, np.zeros_like(dist_coeffs))[0].reshape(-1, 1)

        y_x, y_y, y_z = y[0,0], y[1,0], y[2,0]

        # Commonly used derivatives
        d_xit_d_y = np.array([
            [1/y_z, 0, -y_x/(y_z**2)],
            [0, 1/y_z, -y_y/(y_z**2)],
            [0, 0, 0]
        ])

        d_y_d_Pci = camera_matrix

        d_Pci_d_P1 = R_1_Ci_T

        # Step 4: Compute the jacobian with respect to the tag rotation
        # Convert to euler angles
        r, p, y = rpy_from_dcm(R_1_At).flatten()
        # Get the individual rotation matrices
        R_At_z = dcm_from_rpy(np.array([0, 0, y]))
        R_At_y = dcm_from_rpy(np.array([0, p, 0]))
        R_At_x = dcm_from_rpy(np.array([r, 0, 0]))

        d_P1_d_yaw    = X_bar_z @ R_At_z  @ R_At_y  @ R_At_x @ corner_pos
        d_P1_d_pitch  = R_At_z  @ X_bar_y @ R_At_y  @ R_At_x @ corner_pos
        d_P1_d_roll   = R_At_z  @ R_At_y  @ X_bar_x @ R_At_x @ corner_pos

        d_xit_d_yaw   = d_xit_d_y @ d_y_d_Pci @ d_Pci_d_P1 @ d_P1_d_yaw
        d_xit_d_pitch = d_xit_d_y @ d_y_d_Pci @ d_Pci_d_P1 @ d_P1_d_pitch
        d_xit_d_roll  = d_xit_d_y @ d_y_d_Pci @ d_Pci_d_P1 @ d_P1_d_roll
        
        # corner_tag_rotational_jacobian = np.hstack((d_xit_d_yaw, d_xit_d_pitch, d_xit_d_roll))
        corner_tag_rotational_jacobian = np.hstack((d_xit_d_roll, d_xit_d_pitch, d_xit_d_yaw))

        # Step 6: Compute the jacobian with respect to the tag translation
        d_P1_d_t_At = np.eye(3)
        d_xit_d_t_At = d_xit_d_y @ d_y_d_Pci @ d_Pci_d_P1 @ d_P1_d_t_At

        # Step 7: Construct this corner's tag jacobian
        corner_tag_jacobian = np.hstack((corner_tag_rotational_jacobian, d_xit_d_t_At))
        # corner_tag_jacobian = np.hstack((np.zeros_like(corner_tag_rotational_jacobian), d_xit_d_t_At))
        # The last row corresponds to the z coordinate of the pixel coordinate which is always 0 so we can just delete it
        corner_tag_jacobian = np.delete(corner_tag_jacobian, 2, axis=0)
        
        # Step 8: Compute the jacobian with respect to the camera rotation
        # Convert to euler angles
        r, p, y = rpy_from_dcm(R_1_Ci).flatten()
        # Get the individual rotation matrices. Because we are dealing with the inverse of the rotation matrix, we negate the angles.
        R_Ci_z = dcm_from_rpy(np.array([0, 0, -y]))
        R_Ci_y = dcm_from_rpy(np.array([0, -p, 0]))
        R_Ci_x = dcm_from_rpy(np.array([-r, 0, 0]))

        d_Pci_d_yaw   = -R_Ci_x  @ R_Ci_y  @ X_bar_z @ R_Ci_z @ (p_1 - t_1_Ci)
        d_Pci_d_pitch = -R_Ci_x  @ X_bar_y @ R_Ci_y  @ R_Ci_z @ (p_1 - t_1_Ci)
        d_Pci_d_roll  = -X_bar_x @ R_Ci_x  @ R_Ci_y  @ R_Ci_z @ (p_1 - t_1_Ci)

        d_xit_d_yaw   = d_xit_d_y @ d_y_d_Pci @ d_Pci_d_yaw
        d_xit_d_pitch = d_xit_d_y @ d_y_d_Pci @ d_Pci_d_pitch
        d_xit_d_roll  = d_xit_d_y @ d_y_d_Pci @ d_Pci_d_roll

        # corner_camera_rotational_jacobian = np.hstack((d_xit_d_yaw, d_xit_d_pitch, d_xit_d_roll))
        corner_camera_rotational_jacobian = np.hstack((d_xit_d_roll, d_xit_d_pitch, d_xit_d_yaw))

        # Step 9: Compute the jacobian with respect to the camera translation
        d_Pci_d_t_Ci = -R_1_Ci_T

        d_xit_d_t_Ci = d_xit_d_y @ d_y_d_Pci @ d_Pci_d_t_Ci

        # Step 10: Construct this corner's camera jacobian
        corner_camera_jacobian = np.hstack((corner_camera_rotational_jacobian, d_xit_d_t_Ci))
        # corner_camera_jacobian = np.hstack((np.zeros_like(corner_camera_rotational_jacobian), d_xit_d_t_Ci))
        # corner_camera_jacobian = np.hstack((corner_camera_rotational_jacobian, np.zeros_like(d_xit_d_t_Ci)))

        # The last row corresponds to the z coordinate of the pixel coordinate which is always 0 so we can just delete it
        corner_camera_jacobian = np.delete(corner_camera_jacobian, 2, axis=0)
        
        # Step 11: Add this corner's jacobians to the total jacobians
        # tag_jacobian[2*corner_index:2*corner_index+2,:] = corner_tag_jacobian
        # camera_jacobian[2*corner_index:2*corner_index+2,:] = corner_camera_jacobian
        # expected_pixels[2*corner_index:2*corner_index+2,:] = x_it_cv2
        tag_jacobian.append(corner_tag_jacobian)
        camera_jacobian.append(corner_camera_jacobian)
        expected_pixels.append(x_it_cv2)
        # expected_pixels.append(x_it)
    tag_jacobian = np.vstack(tag_jacobian)
    camera_jacobian = np.vstack(camera_jacobian)
    expected_pixels = np.vstack(expected_pixels)
    
    # Step 12: Construct the full jacobian
    # We need to insert the tag at index 6*tag_index and the camera at index 6*total_features + 6*camera_index
    total_jacobian = np.zeros((2*len(corners), 6*total_features + 6*total_cameras), dtype=np.float64)
    if include_tag_jac:
        total_jacobian[:,6*tag_index:6*tag_index+6] = tag_jacobian
    if include_camera_jac:
        total_jacobian[:,6*total_features + 6*camera_index:6*total_features + 6*camera_index+6] = camera_jacobian
    return total_jacobian, expected_pixels
    
        

def find_jacobian(tag_px_positions, tag_poses, camera_poses, tag_corners_mm_Ai, camera_matrix, dist_coeffs, include_tag_jac=True, include_camera_jac=True):
    """
    Computes the jacobian of the error function for bundle adjustment.
    Stacking many of these jacobians gives the full jacobian for bundle adjustment.

    Each tag has 6 parameters and each camera has 6 parameters.
    Each tag has n corners and each corner has 2 pixel coordinates.
    Therefore, the size of the jacobian is 2nx(6*total_tags + 6*total_cameras). But only 12 of the columns are nonzero.
    """
    tag_ids = sorted(tag_poses.keys())
    camera_indices = range(len(camera_poses))
    partial_jacobians = []
    partial_expected_pixels = []
    partial_true_pixels = []
    for tag_index, tag_id in enumerate(tag_ids):
        # Loop through the images and check if the tag is visible
        for camera_index in camera_indices:
            if tag_px_positions[tag_id][camera_index] is not None:
                tag_pose = tag_poses[tag_id]
                camera_pose = camera_poses[camera_index]
                partial_jacobian, partial_expected_pixel = find_cam_tag_jacobian(tag_index, camera_index, len(tag_ids), len(camera_poses), tag_pose, camera_pose, tag_corners_mm_Ai, camera_matrix, dist_coeffs, include_tag_jac=include_tag_jac, include_camera_jac=include_camera_jac)
                partial_jacobians.append(partial_jacobian)
                partial_expected_pixels.append(partial_expected_pixel)
                partial_true_pixels.append(tag_px_positions[tag_id][camera_index].reshape(-1, 1))
    # Vertically stack the partial jacobians
    jacobian = np.vstack(partial_jacobians)
    expected_pixels = np.vstack(partial_expected_pixels)
    true_pixels = np.vstack(partial_true_pixels)
    # Translate to sparse matrix
    # jacobian = sparse.csr_matrix(jacobian)
    return jacobian, expected_pixels, true_pixels

def params_from_T(T):
    """
    Extracts the parameters from a transformation matrix.
    """
    rpy = rpy_from_dcm(T[:3,:3])
    t = T[:3,3].reshape(-1, 1)
    return np.vstack((rpy, t))

def T_from_params(params):
    """
    Constructs a transformation matrix from the parameters.
    """
    rpy = params[:3,:].reshape(-1, 1)
    t = params[3:,:].reshape(-1, 1)
    R = dcm_from_rpy(rpy)
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = t.flatten()
    return T


def bundle_adjust(tag_px_positions_0, tag_poses_0, camera_poses_0, tag_corners_mm_Ai, camera_matrix, dist_coeffs, base_frame_tag_id=1, use_scipy=True):
    """
    Performs bundle adjustment to refine the estimated tag poses and camera poses.
    Each pixel feature corresponds to the corner as defined by get_tag_corners_mm() of the same index.
    We use a least squares loss of the reprojection error of each pixel feature.

    tag_px_positions_0: Dictionary mapping from tag ID to a list of pixel positions of that tag in each image.
    tag_poses_0: Dictionary mapping from tag ID to a list of poses of that tag in each image. The pose is None if the tag is not visible in the image.
    camera_poses_0: List of camera poses for each image.
    tag_size: The size of the tag in mm.
    camera_matrix: 3x3 numpy array of the camera intrinsic parameters.
    dist_coeffs: 5x1 numpy array of the camera distortion coefficients.
    Returns:
    tag_poses: Dictionary mapping from tag ID to the refined pose
    camera_poses: List of camera poses for each image.
    """
    if not use_scipy:
        # Optimization test 1: Gauss-Newton
        # Construct the parameters vector. This should have size 6*total_tags + 6*total_cameras
        params = np.zeros((6*len(tag_poses_0) + 6*len(camera_poses_0), 1), dtype=np.float64)
        tag_ids = sorted(tag_poses_0.keys())
        for tag_index, tag_id in enumerate(tag_ids):
            params[6*tag_index:6*tag_index+6,:] = params_from_T(tag_poses_0[tag_id])
        for camera_index, camera_pose in enumerate(camera_poses_0):
            params[6*len(tag_poses_0) + 6*camera_index:6*len(tag_poses_0) + 6*camera_index+6,:] = params_from_T(camera_pose)

        prev_params = params
        it = 1
        while True:
            prev_params = params

            # Construct the new tag poses and camera poses
            tag_poses = {}
            for tag_index, tag_id in enumerate(tag_ids):
                tag_poses[tag_id] = T_from_params(params[6*tag_index:6*tag_index+6,:])
            camera_poses = []
            for camera_index in range(len(camera_poses_0)):
                camera_poses.append(T_from_params(params[6*len(tag_poses_0) + 6*camera_index:6*len(tag_poses_0) + 6*camera_index+6,:]))

            # Compute the jacobian
            jacobian, expected_pixels, true_pixels = find_jacobian(tag_px_positions_0, tag_poses, camera_poses, tag_corners_mm_Ai, camera_matrix, dist_coeffs, include_tag_jac=True, include_camera_jac=True)

            # Compute the error
            # error = expected_pixels - true_pixels
            error = true_pixels - expected_pixels

            # Compute the update
            H = jacobian.T @ jacobian
            b = jacobian.T @ error
            # Compute the update using a linear solver
            delta = np.linalg.lstsq(H, b, rcond=None)[0]
            
            # Update the parameters
            params = params + delta

            # Check for convergence
            delta_mag = np.linalg.norm(delta)
            if delta_mag < 1e-6:
                break

            print(f"Iteration {it}: delta = {round(delta_mag, 3)}, error = {round(np.linalg.norm(error), 3)}")
            it += 1

        # Construct the new tag poses and camera poses
        tag_poses = {}
        for tag_index, tag_id in enumerate(tag_ids):
            tag_poses[tag_id] = T_from_params(params[6*tag_index:6*tag_index+6,:])
        camera_poses = []
        for camera_index in range(len(camera_poses_0)):
            camera_poses.append(T_from_params(params[6*len(tag_poses_0) + 6*camera_index:6*len(tag_poses_0) + 6*camera_index+6,:]))

        # Rotate all poses so that tag id 1 is at 0,0,0 and has no rotation
        # We can do this by multiplying everything by the inverse of the pose of tag 1
        tag_1_pose = tag_poses[1]
        tag_1_pose_inv = np.linalg.inv(tag_1_pose)
        for tag_id in tag_poses:
            tag_poses[tag_id] = tag_1_pose_inv @ tag_poses[tag_id]
        for camera_index in range(len(camera_poses)):
            camera_poses[camera_index] = tag_1_pose_inv @ camera_poses[camera_index]

        return tag_poses, camera_poses
    else:
        # Using scipy's least squares solver
        # Construct the parameters vector. This should have size 6*total_tags + 6*total_cameras
        params = np.zeros(6*len(tag_poses_0) + 6*len(camera_poses_0), dtype=np.float64)
        tag_ids = sorted(tag_poses_0.keys())
        for tag_index, tag_id in enumerate(tag_ids):
            params[6*tag_index:6*tag_index+6] = params_from_T(tag_poses_0[tag_id]).flatten()
        for camera_index, camera_pose in enumerate(camera_poses_0):
            params[6*len(tag_poses_0) + 6*camera_index:6*len(tag_poses_0) + 6*camera_index+6] = params_from_T(camera_pose).flatten()
        def error_func(params):
            # Construct the new tag poses and camera poses
            tag_poses = {}
            for tag_index, tag_id in enumerate(tag_ids):
                tag_poses[tag_id] = T_from_params(params[6*tag_index:6*tag_index+6].reshape(-1, 1))
            camera_poses = []
            for camera_index in range(len(camera_poses_0)):
                camera_poses.append(T_from_params(params[6*len(tag_poses_0) + 6*camera_index:6*len(tag_poses_0) + 6*camera_index+6].reshape(-1, 1)))

            # Compute the jacobian
            jacobian, expected_pixels, true_pixels = find_jacobian(tag_px_positions_0, tag_poses, camera_poses, tag_corners_mm_Ai, camera_matrix, dist_coeffs, include_tag_jac=True, include_camera_jac=True)

            # Compute the error
            error = expected_pixels - true_pixels

            return error.flatten()

        def jacobian_func(params):
            # Construct the new tag poses and camera poses
            tag_poses = {}
            for tag_index, tag_id in enumerate(tag_ids):
                tag_poses[tag_id] = T_from_params(params[6*tag_index:6*tag_index+6].reshape(-1, 1))
            camera_poses = []
            for camera_index in range(len(camera_poses_0)):
                camera_poses.append(T_from_params(params[6*len(tag_poses_0) + 6*camera_index:6*len(tag_poses_0) + 6*camera_index+6].reshape(-1, 1)))

            # Compute the jacobian
            jacobian, expected_pixels, true_pixels = find_jacobian(tag_px_positions_0, tag_poses, camera_poses, tag_corners_mm_Ai, camera_matrix, dist_coeffs, include_tag_jac=True, include_camera_jac=True)

            return jacobian

        # Call the solver
        result = least_squares(error_func, params, jac=jacobian_func, verbose=2)
        # result = least_squares(error_func, params, jac='3-point', verbose=2)
        
        # Construct the new tag poses and camera poses
        tag_poses = {}
        for tag_index, tag_id in enumerate(tag_ids):
            tag_poses[tag_id] = T_from_params(result.x[6*tag_index:6*tag_index+6].reshape(-1, 1))
        camera_poses = []
        for camera_index in range(len(camera_poses_0)):
            camera_poses.append(T_from_params(result.x[6*len(tag_poses_0) + 6*camera_index:6*len(tag_poses_0) + 6*camera_index+6].reshape(-1, 1)))

        # Rotate all poses so that tag id 1 is at 0,0,0 and has no rotation
        # We can do this by multiplying everything by the inverse of the pose of tag 1
        tag_1_pose = tag_poses[base_frame_tag_id]
        tag_1_pose_inv = np.linalg.inv(tag_1_pose)
        for tag_id in tag_poses:
            tag_poses[tag_id] = tag_1_pose_inv @ tag_poses[tag_id]
        for camera_index in range(len(camera_poses)):
            camera_poses[camera_index] = tag_1_pose_inv @ camera_poses[camera_index]

        return tag_poses, camera_poses



    