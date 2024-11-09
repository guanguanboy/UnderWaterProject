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

def estimate_mbr_in_sqm(ply_file, intrinsics):
    """
    估计点云场景的最小外接矩形的长、宽及面积（平方米）。
    
    参数:
    - ply_file: str, 点云文件的路径 (.ply 文件)。
    - intrinsics: 3x3 numpy array, 相机内参矩阵。
    
    返回:
    - length_m: float, 外接矩形的长度（米）。
    - width_m: float, 外接矩形的宽度（米）。
    - area_sqm: float, 外接矩形的面积（平方米）。
    """
    # 加载点云数据
    points = load_ply_points(ply_file)

    # 从相机内参矩阵中获取焦距和光心
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # 计算平均深度
    mean_depth = np.mean(points[:, 2])

    # 计算像素在真实世界中的大小
    pixel_size = mean_depth / fx  # 或使用 fy，假设 fx 和 fy 近似相等

    # 投影到图像平面
    projected_points = np.zeros((points.shape[0], 2))
    projected_points[:, 0] = (points[:, 0] * fx) / points[:, 2] + cx
    projected_points[:, 1] = (points[:, 1] * fy) / points[:, 2] + cy

    # 计算最小外接矩形的长和宽
    min_x, max_x = np.min(projected_points[:, 0]), np.max(projected_points[:, 0])
    min_y, max_y = np.min(projected_points[:, 1]), np.max(projected_points[:, 1])
    
    length_pixel = max_x - min_x  # 外接矩形的长度（像素）
    width_pixel = max_y - min_y   # 外接矩形的宽度（像素）

    # 将长度和宽度换算为米
    length_m = length_pixel * pixel_size
    width_m = width_pixel * pixel_size

    # 计算面积（平方米）
    area_sqm = length_m * width_m

    return length_m, width_m, area_sqm

# 示例用法
# 示例相机内参矩阵
#intrinsics = np.array([[1518.50, 0, 685.90], [0, 1519.36, 450.59], [0, 0, 1]]) #curasao_new
#intrinsics = np.array([[1511.44, 0, 686.68], [0, 1511.44, 455.89], [0, 0, 1]]) #output_JapaneseGradens-RedSea
intrinsics = np.array([[1961.25, 0,  891.32], [0, 1958.02, 583.51], [0, 0, 1]]) #panama
# 使用示例点云文件
#ply_file = "curasao.ply"  # 替换为实际 .ply 文件路径
#ply_file = "output_IUI3-RedSea.ply" 
ply_file = "output_Panama.ply" 
#ply_file = "output_JapaneseGradens-RedSea.ply" 
length_m, width_m, area_sqm = estimate_mbr_in_sqm(ply_file, intrinsics)
print("外接矩形的长度（米）:", length_m)
print("外接矩形的宽度（米）:", width_m)
print("外接矩形的面积（平方米）:", area_sqm)
