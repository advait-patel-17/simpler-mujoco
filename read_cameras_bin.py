import sqlite3
import numpy as np

def get_intrinsics_from_colmap_db(db_path):
    """Extract camera parameters from COLMAP database file."""
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    
    cursor.execute("SELECT camera_id, model, params, width, height FROM cameras;")
    cameras = {}
    
    for row in cursor:
        camera_id, model, params, width, height = row
        
        # Parse the parameters blob
        params = np.frombuffer(params, dtype=np.float64)
        
        # Map model ID to name
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
        }.get(model, f"Unknown model {model}")
        
        cameras[camera_id] = {
            'model_id': model,
            'model_name': model_name,
            'width': width,
            'height': height,
            'params': params
        }
        
        print(f"Camera {camera_id} ({model_name}): {width}x{height}")
        print(f"All parameters: {params}")
        
        # For OPENCV model
        if model_name == "OPENCV":
            # Parameters are: fx, fy, cx, cy, k1, k2, p1, p2
            if len(params) >= 8:
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
                k1, k2, p1, p2 = params[4], params[5], params[6], params[7]
                
                # Create intrinsic matrix
                K = np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ])
                
                print("Intrinsic Matrix K:")
                print(K)
                print("\nDistortion Parameters:")
                print(f"Radial: k1={k1}, k2={k2}")
                print(f"Tangential: p1={p1}, p2={p2}")
                
                # Print in format for common computer vision libraries
                print("\nOpenCV format:")
                print(f"K = np.array([")
                print(f"    [{fx}, 0, {cx}],")
                print(f"    [0, {fy}, {cy}],")
                print(f"    [0, 0, 1]")
                print(f"])")
                print(f"dist_coeffs = np.array([{k1}, {k2}, {p1}, {p2}, 0])")
                
                # For 3D Gaussian Splatting
                print("\nFor 3D Gaussian Splatting (if needed):")
                print(f"fx: {fx}")
                print(f"fy: {fy}")
                print(f"cx: {cx}")
                print(f"cy: {cy}")
                print(f"width: {width}")
                print(f"height: {height}")
            else:
                print("Warning: Not enough parameters for OPENCV model!")
        
    connection.close()
    return cameras

# Path to your COLMAP database
db_path = "/home/shivansh/Projects/3dgs/outputs/setup_cam_cali/colmap/database.db"
cameras = get_intrinsics_from_colmap_db(db_path)