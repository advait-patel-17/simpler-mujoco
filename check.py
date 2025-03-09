import importlib

def check_opencv_installation(module_name="cv2"):
    """
    Checks if the specified OpenCV module is installed.

    Args:
        module_name (str, optional): The name of the OpenCV module to check 
                                     ("cv2" for opencv-python or 
                                     "cv2.cv2" for opencv-contrib-python). 
                                     Defaults to "cv2".

    Returns:
        bool: True if the module is installed, False otherwise.
    """
    try:
        importlib.import_module(module_name)
        print(f"{module_name} is installed.")
        return True
    except ImportError:
        print(f"{module_name} is not installed.")
        return False

# Check for opencv-python
check_opencv_installation()

# Check for opencv-contrib-python 
# (Note: the correct way to import contrib modules changed after OpenCV 3.x)
check_opencv_installation("cv2.cv2")