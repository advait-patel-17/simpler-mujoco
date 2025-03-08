import numpy as np
from scipy.spatial.transform import Rotation
import os

def transform_camera_params(matrix):
    # Create a transformation matrix that flips x and y axes
    # flip_matrix = np.array([
    #     [0, 1, 0],
    #     [-1, 0, 0],
    #     [0, 0, 1]
    # ])

    rot_1 = np.array([
        [0, -1, 0, -0.2],
        [1, 0, 0, 0.16],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    pre_rot = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])

    flipped_homogen = rot_1 @ matrix @ pre_rot

    # # Extract and flip the position (last column of the matrix)
    # position = matrix[:3, 3]
    # flipped_position = flip_matrix @ position
    
    # # Apply the flip to the rotation matrix
    # flipped_rotation =  flip_2 @ flip_matrix @ matrix[:3, :3]
    flipped_rotation = flipped_homogen[:3, :3]
    flipped_position = flipped_homogen[:3, 3]
    # Convert to quaternion using scipy
    r = Rotation.from_matrix(flipped_rotation)
    quat = r.as_quat()  # returns in x, y, z, w format
    new_quat = np.array([quat[3], quat[0], quat[1], quat[2]])  # convert to w, x, y, z format
    
    return flipped_position, new_quat

def process_extrinsics():
    extrinsics_dir = './cam_extrinsics/'
    
    if not os.path.exists(extrinsics_dir):
        print(f"Error: Directory {extrinsics_dir} does not exist!")
        return
    
    npy_files = [f for f in os.listdir(extrinsics_dir) if f.endswith('.npy')]
    
    print("Camera parameters after flipping x and y axes:\n")
    
    for file_name in sorted(npy_files):
        file_path = os.path.join(extrinsics_dir, file_name)
        try:
            data = np.load(file_path)
            # not having the inverse clumps all the cameras together in the center for some reason
            # ok so here's why we need inverse - camera extrinsics are the transformation from camera to world
            # but we need the transformation from world to camera
            data = np.linalg.inv(data)
            position, quat = transform_camera_params(data)
            print(f'    <camera name="cam_{file_name[:-4]}" pos="{position[0]} {position[1]} {position[2]}" ')
            print(f'      quat="{quat[0]} {quat[1]} {quat[2]} {quat[3]}" mode="fixed" />')
        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")

if __name__ == "__main__":
    process_extrinsics() 