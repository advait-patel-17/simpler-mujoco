import struct
import numpy as np

def read_cameras_binary(path_to_model_file):
    """
    Read COLMAP camera binary file.
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = struct.unpack('<Q', fid.read(8))[0]
        for _ in range(num_cameras):
            camera_id = struct.unpack('<i', fid.read(4))[0]
            model_id = struct.unpack('<i', fid.read(4))[0]
            width = struct.unpack('<Q', fid.read(8))[0]
            height = struct.unpack('<Q', fid.read(8))[0]
            
            # The number of parameters depends on the camera model
            num_params = struct.unpack('<i', fid.read(4))[0]
            params = struct.unpack('<' + 'd' * num_params, fid.read(8 * num_params))
            
            cameras[camera_id] = {
                'model_id': model_id,
                'width': width,
                'height': height,
                'params': params
            }
            
            # Map common model IDs to their names
            model_name = {
                0: "SIMPLE_PINHOLE",
                1: "PINHOLE",
                2: "SIMPLE_RADIAL",
                3: "RADIAL",
                4: "OPENCV",
                5: "OPENCV_FISHEYE",
                6: "FULL_OPENCV",
                7: "FOV",
                8: "SIMPLE_RADIAL_FISHEYE",
                9: "RADIAL_FISHEYE",
                10: "THIN_PRISM_FISHEYE"
            }.get(model_id, f"Unknown model {model_id}")
            
            cameras[camera_id]['model_name'] = model_name
            
    return cameras

# Example usage
cameras = read_cameras_binary("path/to/cameras.bin")

# Print camera info and intrinsics
for camera_id, camera in cameras.items():
    print(f"Camera ID: {camera_id}")
    print(f"Camera Model: {camera['model_name']}")
    print(f"Width: {camera['width']}, Height: {camera['height']}")
    print(f"Parameters: {camera['params']}")
    
    # Extract intrinsics based on the camera model
    if camera['model_name'] == "SIMPLE_PINHOLE":
        # f, cx, cy
        f, cx, cy = camera['params']
        K = np.array([[f, 0, cx],
                      [0, f, cy],
                      [0, 0, 1]])
        print("Intrinsic Matrix K:")
        print(K)
    elif camera['model_name'] == "PINHOLE":
        # fx, fy, cx, cy
        fx, fy, cx, cy = camera['params']
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]])
        print("Intrinsic Matrix K:")
        print(K)