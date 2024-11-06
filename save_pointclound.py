import numpy as np
import open3d as o3d
from PIL import Image

def depth_to_point_cloud(depth_map, intrinsics):
    """
    Convert a depth map to a 3D point cloud using camera intrinsics.
    
    :param depth_map: The depth map (2D array) of the image.
    :param intrinsics: The camera intrinsic matrix (3x3 matrix).
    :return: The point cloud in Open3D format.
    """
    # Get the dimensions of the depth map
    height, width = depth_map.shape

    # Generate a grid of pixel coordinates
    i, j = np.meshgrid(np.arange(width), np.arange(height))

    # Flatten the arrays
    i = i.flatten()
    j = j.flatten()
    depth = depth_map.flatten()

    # Convert pixel coordinates to camera coordinates
    x = (i - intrinsics[0, 2]) * depth / intrinsics[0, 0]
    y = (j - intrinsics[1, 2]) * depth / intrinsics[1, 1]
    z = depth

    # Stack the coordinates to form the 3D points
    points = np.vstack((x, y, z)).T

    # Remove points with zero depth
    valid_points = points[depth > 0]

    # Convert to Open3D point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(valid_points)

    return point_cloud

# Example usage:
# Define the camera intrinsic matrix (fx, fy, cx, cy)
intrinsics = np.array([[525, 0, 319.5],  # fx, 0, cx
                       [0, 525, 239.5],  # 0, fy, cy
                       [0, 0, 1]])       # 0, 0, 1

# Load depth map from PNG file (make sure it is in 16-bit grayscale)
depth_image = Image.open('depth/00002.png')  # Open the PNG image
depth_map = np.array(depth_image)         # Convert to NumPy array

# Normalize depth_map if it is in 16-bit format (values range from 0 to 65535)
# You can skip this step if your depth map is already in meters
depth_map = depth_map.astype(np.float32) / 1000.0  # Assuming depth is in mm, convert to meters

# Convert the depth map to point cloud
point_cloud = depth_to_point_cloud(depth_map, intrinsics)

# Save the point cloud to a PLY file
o3d.io.write_point_cloud("output_point_cloud.ply", point_cloud)

print("Point cloud saved to output_point_cloud.ply")
