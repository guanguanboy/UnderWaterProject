import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def convert_to_16bit_grayscale(image):
    # 确保图像模式是RGBA
    image = image.convert('RGBA')
    
    # 创建一个新的16-bit灰度图像
    gray_image = Image.new('I;16', image.size, 0)
    
    # 遍历每个像素并计算16-bit灰度值
    for x in range(image.width):
        for y in range(image.height):
            rgba = image.getpixel((x, y))
            gray = int((rgba[0] + rgba[1] + rgba[2]) / 3 * 256)
            gray_image.putpixel((x, y), gray)
    
    return gray_image
 

def depth_to_point_cloud(depth_map, intrinsics):
    """
    Convert a depth map to a 3D point cloud using camera intrinsics.
    
    :param depth_map: The depth map (2D array) of the image.
    :param intrinsics: The camera intrinsic matrix (3x3 matrix).
    :return: The point cloud in numpy array format (Nx3).
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

    return valid_points

def save_ply(filename, points):
    """
    Save 3D points to a PLY file.
    
    :param filename: The file name to save the point cloud.
    :param points: The 3D points to save.
    """
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex {}\n".format(len(points)))
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for point in points:
            f.write("{} {} {}\n".format(point[0], point[1], point[2]))

# Example usage:
# Define the camera intrinsic matrix (fx, fy, cx, cy)
intrinsics = np.array([[36, 0, 0],  # fx, 0, cx
                       [0, 24, 0],  # 0, fy, cy
                       [0, 0, 1]])       # 0, 0, 1

# Load depth map from PNG file (make sure it is in 16-bit grayscale)
depth_image = Image.open('Depth/Image0004.png')
depth_image = depth_image.convert('L')  # Open the PNG image
gray_image_16bit = depth_image.convert("I")  # 将L模式转换为32位整数模式

print(gray_image_16bit.size)
print(gray_image_16bit.mode)

depth_map = np.array(gray_image_16bit, dtype=np.uint16)
#depth_image = convert_to_16bit_grayscale(depth_image)
#depth_image = Image.open('depth/00182.png')

#depth_image = depth_image.convert('L')  # Open the PNG image
print(depth_image.mode)
#depth_map = np.array(depth_image)         # Convert to NumPy array
print(depth_map.shape)

# Normalize depth_map if it is in 16-bit format (values range from 0 to 65535)
# You can skip this step if your depth map is already in meters
depth_map = depth_map.astype(np.float32) / 1000.0  # Assuming depth is in mm, convert to meters
#depth_map = depth_map.astype(np.float32)
# Convert the depth map to point cloud
point_cloud = depth_to_point_cloud(depth_map, intrinsics)

# Save the point cloud to a PLY file
save_ply("Image0004.ply", point_cloud)

print("Point cloud saved to output_point_cloud.ply")

# Optional: Visualize using matplotlib (for a simple 3D scatter plot)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
