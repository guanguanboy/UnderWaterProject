import numpy as np
import cv2
import open3d as o3d

# 加载深度图像
depth_image = cv2.imread('depth/00002.png', cv2.IMREAD_UNCHANGED)  # 假设深度图以16位图像存储

# 相机内参矩阵（需要根据实际相机进行调整）
fx = 525.0  # 焦距（单位：像素）
fy = 525.0  # 焦距（单位：像素）
cx = 319.5  # 主点x坐标（单位：像素）
cy = 239.5  # 主点y坐标（单位：像素）

# 生成网格坐标系
height, width = depth_image.shape
x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))

# 使用内参将像素坐标转换为相机坐标
z = depth_image / 1000.0  # 将深度值从毫米转换为米
x_camera = (x - cx) * z / fx
y_camera = (y - cy) * z / fy

# 创建点云
points = np.stack((x_camera, y_camera, z), axis=-1).reshape(-1, 3)

# 过滤掉深度值为0的点（无效点）
points = points[points[:, 2] > 0]

# 使用Open3D将点云可视化
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)

# 可视化
o3d.visualization.draw_geometries([point_cloud])

