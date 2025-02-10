import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from mpl_toolkits.mplot3d import Axes3D
import cv2

class TrajectoryViewer:
    def __init__(self, filepath):
        self.file = h5py.File(filepath, 'r')
        self.current_time_idx = 0
        self.fig = None
        self.axes = None
        
    def print_dataset_info(self, dataset_name):
        """Print detailed information about a specific dataset"""
        dataset = self.file[dataset_name]
        print(f"\nDataset: {dataset_name}")
        print(f"Shape: {dataset.shape}")
        print(f"Type: {dataset.dtype}")
        
        # Print sample data
        data = dataset[:]
        if len(data.shape) == 1:
            print("Sample values (first 5):", data[:5])
        elif len(data.shape) == 2:
            print("Sample values (first row):", data[0])
        
        # Additional information based on dataset type
        if dataset_name.endswith('_extrinsics'):
            print("\nThis is a camera extrinsics matrix (camera pose in world frame)")
            print("First frame transform:")
            print(data[0])
        elif dataset_name.endswith('_intrinsics'):
            print("\nThis is a camera intrinsics matrix (internal camera parameters)")
            print("First frame intrinsics:")
            print(data[0])
        elif 'joint_pos' in dataset_name:
            print("\nThese are joint positions for the robot's 7 joints")
        elif 'ee_pos' in dataset_name:
            print("\nThis is the end-effector position data")
        elif dataset_name == 'stage':
            print("\nThis indicates the stage of the trajectory")
            unique_stages = np.unique(data)
            print(f"Unique stages: {unique_stages}")
        elif dataset_name == 'timestamp':
            print("\nThese are timestamps for each frame")
            print(f"Total duration: {data[-1] - data[0]:.2f} seconds")

    def plot_joint_trajectories(self):
        """Plot all joint trajectories"""
        joint_pos = self.file['observations/joint_pos'][:]
        timestamps = self.file['timestamp'][:]
        
        plt.figure(figsize=(12, 6))
        for i in range(joint_pos.shape[1]):
            plt.plot(timestamps - timestamps[0], joint_pos[:, i], label=f'Joint {i+1}')
        
        plt.xlabel('Time (s)')
        plt.ylabel('Joint Position')
        plt.title('Joint Trajectories')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_end_effector_trajectory(self):
        """Plot end-effector trajectory in 3D"""
        ee_pos = self.file['observations/ee_pos'][:]
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.plot(ee_pos[:, 0], ee_pos[:, 1], ee_pos[:, 2])
        ax.scatter(ee_pos[0, 0], ee_pos[0, 1], ee_pos[0, 2], color='green', label='Start')
        ax.scatter(ee_pos[-1, 0], ee_pos[-1, 1], ee_pos[-1, 2], color='red', label='End')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('End-Effector Trajectory')
        plt.legend()
        plt.show()

    def view_camera_data(self, camera_idx=0, frame_idx=0):
        """View color and depth data from a specific camera at a specific frame"""
        # Color image
        color = self.file[f'observations/images/camera_{camera_idx}_color'][frame_idx]
        depth = self.file[f'observations/images/camera_{camera_idx}_depth'][frame_idx]
        
        # Normalize depth for visualization
        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_norm = depth_norm.astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        ax1.imshow(color)
        ax1.set_title(f'Color Image - Camera {camera_idx}')
        ax2.imshow(depth_colored)
        ax2.set_title(f'Depth Image - Camera {camera_idx}')
        plt.show()

    def interactive_exploration(self):
        """Interactive menu for exploring different aspects of the data"""
        while True:
            print("\nWhat would you like to explore?")
            print("1. View dataset information")
            print("2. Plot joint trajectories")
            print("3. Plot end-effector trajectory")
            print("4. View camera data")
            print("5. Exit")
            
            choice = input("Enter your choice (1-5): ")
            
            if choice == '1':
                print("\nAvailable datasets:")
                for name, item in self.file.items():
                    if isinstance(item, h5py.Dataset):
                        print(f"- {name}")
                    elif isinstance(item, h5py.Group):
                        self._print_group_datasets(item, prefix="  ")
                
                dataset_name = input("\nEnter dataset name to explore: ")
                try:
                    self.print_dataset_info(dataset_name)
                except KeyError:
                    print("Dataset not found!")
            
            elif choice == '2':
                self.plot_joint_trajectories()
            
            elif choice == '3':
                self.plot_end_effector_trajectory()
            
            elif choice == '4':
                camera_idx = int(input("Enter camera index (0-3): "))
                frame_idx = int(input(f"Enter frame index (0-{len(self.file['timestamp'])-1}): "))
                self.view_camera_data(camera_idx, frame_idx)
            
            elif choice == '5':
                break
            
            else:
                print("Invalid choice!")

    def _print_group_datasets(self, group, prefix=""):
        """Helper function to print datasets in a group"""
        for name, item in group.items():
            if isinstance(item, h5py.Dataset):
                print(f"{prefix}- {group.name}/{name}")
            elif isinstance(item, h5py.Group):
                self._print_group_datasets(item, prefix=prefix+"  ")

    def close(self):
        """Close the HDF5 file"""
        self.file.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Explore robot trajectory data from HDF5 file')
    parser.add_argument('filepath', type=str, help='Path to HDF5 file')
    args = parser.parse_args()
    
    viewer = TrajectoryViewer(args.filepath)
    viewer.interactive_exploration()
    viewer.close()

if __name__ == "__main__":
    main()