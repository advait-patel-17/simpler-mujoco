import cv2
import os
import numpy as np
import time
import warnings


def calibrate_extrinsics(visualize=True, board_size=(6,9), squareLength=0.03, markerLength=0.022):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    board = cv2.aruco.CharucoBoard(
        size=board_size,
        squareLength=squareLength,
        markerLength=markerLength,
        dictionary=dictionary,
    )
    # TODO get intrinsics
    K = np.array([
    [1536.0, 0, 640.0],
    [0, 1536.0, 360.0],
    [0, 0, 1]
    ])
    dist_coef = np.array([0.0, 0.0, 0.0, 0.0, 0])
    intrinsic_matrix = K

    R_gripper2base = []
    t_gripper2base = []
    R_board2cam = []
    t_board2cam = []
    rgbs = []
    depths = []
    point_list = []
    masks = []


    # Calculate the markers
    # out = self.ring_buffer.get()
    # # points, colors, depth_img, mask = self.camera.get_observations()
    # colors = out['color']
    # calibration_img = colors.copy()
    # TODO: get the calibration image
    img_path = "/home/shivansh/Projects/3dgs/outputs/setup_cam_cali/images/frame_00112.png"
    calibration_img = cv2.imread(img_path)
    cv2.imshow(f"calibration_img", calibration_img)
    cv2.waitKey(1)
    # import pdb
    # pdb.set_trace()
    corners, ids, rejected = cv2.aruco.detectMarkers(
        calibration_img, 
        dictionary,
        parameters=cv2.aruco.DetectorParameters()
    )
    if len(corners) == 0:
        warnings.warn('no markers detected')
        return

    # import pdb
    # pdb.set_trace()

    # Calculate the charuco corners
    retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        markerCorners=corners,
        markerIds=ids,
        image=calibration_img,
        board=board,
        cameraMatrix=intrinsic_matrix,
        distCoeffs=dist_coef,
    )
    # all_charuco_corners.append(charuco_corners)
    # all_charuco_ids.append(charuco_ids)

    if charuco_corners is None:
        warnings.warn('no charuco corners detected')
        return

    print("number of corners: ", len(charuco_corners))

    visualize = True
    if visualize:
        cv2.aruco.drawDetectedCornersCharuco(
            image=calibration_img,
            charucoCorners=charuco_corners,
            charucoIds=charuco_ids,
        )
        cv2.imshow(f"calibration", calibration_img)
        cv2.waitKey(1)

    rvec = None
    tvec = None
    retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
        charuco_corners,
        charuco_ids,
        board,
        intrinsic_matrix,
        dist_coef,
        rvec,
        tvec,
    )
    
    if not retval:
        warnings.warn('pose estimation failed')
        return

    reprojected_points, _ = cv2.projectPoints(
        board.getChessboardCorners()[charuco_ids, :],
        rvec,
        tvec,
        intrinsic_matrix,
        dist_coef,
    )

    # Reshape for easier handling
    reprojected_points = reprojected_points.reshape(-1, 2)
    charuco_corners = charuco_corners.reshape(-1, 2)

    # Calculate the error
    error = np.sqrt(
        np.sum((reprojected_points - charuco_corners) ** 2, axis=1)
    ).mean()

    print("Reprojection Error:", error)


    print("retval: ", retval)
    print("rvec: ", rvec)
    print("tvec: ", tvec)
    if not retval:
        raise ValueError("pose estimation failed")
    R_board2cam=cv2.Rodrigues(rvec)[0]
    t_board2cam=tvec[:, 0]
    print("R_board2cam: ", R_board2cam)
    print("t_board2cam: ", t_board2cam)

    tf = np.eye(4)
    tf[:3, :3] = R_board2cam
    tf[:3, 3] = t_board2cam
    
    tf_world2board = np.eye(4)
    tf_world2board[:3, :3] = np.array([[0, -1, 0],
                                        [-1, 0, 0],
                                        [0, 0, -1]])
    tf = tf @ tf_world2board
    # #make tf as list
    # tf = tf.tolist()
    # print("tf: ", tf)
    print("tf: ", tf)
    # Save the transformation matrix to the current working directory
    np.save('camera_extrinsics.npy', tf)
    print(f"Saved transformation matrix to {os.path.join(os.getcwd(), 'camera_extrinsics.npy')}")

calibrate_extrinsics()