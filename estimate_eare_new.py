import numpy as np
from plyfile import PlyData
from scipy.spatial import ConvexHull

def load_ply_points(ply_file):
    """
    使用 plyfile 库读取 .ply 文件中的点云数据。
    
    参数:
    - ply_file: str, 点云文件的路径 (.ply 文件)。
    
    返回:
    - points: numpy array of shape (N, 3), 点云坐标。
    """
    # 使用 plyfile 读取 .ply 文件
    ply_data = PlyData.read(ply_file)
    points = np.vstack([ply_data['vertex']['x'], 
                        ply_data['vertex']['y'], 
                        ply_data['vertex']['z']]).T
    return points

def estimate_scene_area_in_sqm(ply_file, intrinsics):
    """
    估计点云场景所占的最大面积，以平方米为单位。
    
    参数:
    - ply_file: str, 点云文件的路径 (.ply 文件)。
    - intrinsics: 3x3 numpy array, 相机内参矩阵。
    
    返回:
    - area_sqm: float, 点云投影在图像平面上的最大面积（平方米）。
    """
    # 加载点云数据
    points = load_ply_points(ply_file)

    # 从相机内参矩阵中获取焦距和光心
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # 计算最大深度
    mean_depth = np.max(points[:, 2])

    # 计算像素在真实世界中的大小
    pixel_size = mean_depth / fx  # 或使用 fy，假设 fx 和 fy 近似相等

    # 投影到图像平面
    projected_points = np.zeros((points.shape[0], 2))
    projected_points[:, 0] = (points[:, 0] * fx) / points[:, 2] + cx
    projected_points[:, 1] = (points[:, 1] * fy) / points[:, 2] + cy

    # 计算凸包以估计面积
    hull = ConvexHull(projected_points)
    pixel_area = hull.area

    # 换算为平方米
    area_sqm = pixel_area * (pixel_size ** 2)

    return area_sqm

# 示例用法
# 示例相机内参矩阵
intrinsics = np.array([[1518.50, 0, 685.90], [0, 1519.36, 450.59], [0, 0, 1]])

# 使用示例点云文件
ply_file = "curasao.ply"  # 替换为实际 .ply 文件路径
area_sqm = estimate_scene_area_in_sqm(ply_file, intrinsics)
print("场景的最大投影面积 (平方米):", area_sqm)
