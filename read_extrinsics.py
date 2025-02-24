import os
import numpy as np

def read_extrinsics():
    # Path to the directory containing extrinsics
    extrinsics_dir = './cam_extrinsics/'
    
    # Check if directory exists
    if not os.path.exists(extrinsics_dir):
        print(f"Error: Directory {extrinsics_dir} does not exist!")
        return
    
    # Get all .npy files in the directory
    npy_files = [f for f in os.listdir(extrinsics_dir) if f.endswith('.npy')]
    
    if not npy_files:
        print("No .npy files found in the directory!")
        return
    
    # Read and print each .npy file
    for file_name in sorted(npy_files):
        file_path = os.path.join(extrinsics_dir, file_name)
        try:
            data = np.load(file_path)
            print(f"\n=== Contents of {file_name} ===")
            print(data)
        except Exception as e:
            print(f"Error reading {file_name}: {str(e)}")

if __name__ == "__main__":
    read_extrinsics()
