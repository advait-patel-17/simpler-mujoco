import numpy as np
from scipy.spatial.transform import Rotation
import os

def transform_camera_params(matrix):
    # Create a transformation matrix that flips x and y axes
    flip_matrix = np.array([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
    ])
    
    # Extract and flip the position (last column of the matrix)
    position = matrix[:3, 3]
    flipped_position = flip_matrix @ position
    
    # Apply the flip to the rotation matrix
    flipped_rotation = matrix[:3, :3] @ flip_matrix
    
    # Convert to quaternion using scipy
    r = Rotation.from_matrix(flipped_rotation)
    quat = r.as_quat()  # returns in x, y, z, w format
    
    return flipped_position, quat

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
            position, quat = transform_camera_params(data)
            print(f"Camera {file_name}:")
            print(f'pos="{position[0]} {position[1]} {position[2]}"')
            print(f'quat="{quat[3]} {quat[0]} {quat[1]} {quat[2]}"\n')
        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")

if __name__ == "__main__":
    process_extrinsics() 