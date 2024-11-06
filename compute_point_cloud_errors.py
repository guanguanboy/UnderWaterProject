"""
import numpy as np
from scipy.spatial import cKDTree

# 假设点云 A 和点云 B 是 Numpy 数组，形状为 (N, 3)
point_cloud_A = np.random.rand(100, 3)  # 示例点云 A
point_cloud_B = np.random.rand(100, 3)  # 示例点云 B

# 使用 KDTree 找到点云 A 和 B 中的最近邻点
tree_A = cKDTree(point_cloud_A)
distances, _ = tree_A.query(point_cloud_B)

# 计算平均点对点误差
mean_error = np.mean(distances)
print(f"Mean point-to-point error: {mean_error} meters")
"""
from plyfile import PlyData
import numpy as np
def read_colmap_ply_with_plyfile(file_path):
    """使用plyfile库读取COLMAP导出的PLY文件中的点云数据"""
    plydata = PlyData.read(file_path)
    
    # 获取点云数据（假设是名为'vertex'的元素，包含x, y, z, r, g, b）
    vertex_data = plydata['vertex']
    
    # 提取x, y, z, r, g, b坐标和颜色
    #points = np.vstack([vertex_data['x'], vertex_data['y'], vertex_data['z'], 
    #                    vertex_data['red'], vertex_data['green'], vertex_data['blue']]).T

    # 提取x, y, z, r, g, b坐标和颜色
    points = np.vstack([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T
    return points

def read_ply_points(file_path):
    """读取PLY文件中的点云数据"""
    with open(file_path, 'r') as file:
        # 读取文件头部，直到遇到end_header
        line = file.readline()
        while line.strip() != 'end_header':
            line = file.readline()
        
        # 读取点云数据部分，存储为一个点的列表
        points = []
        for line in file:
            coords = list(map(float, line.strip().split()))
            points.append(coords[:3])  # 假设每个点的坐标是前三个值 (x, y, z)
        return np.array(points)

def read_colmap_ply(file_path):
    """读取COLMAP导出的PLY文件中的点云数据"""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        # 跳过文件头部，直到遇到end_header
        line = file.readline()
        while line.strip() != 'end_header':
            line = file.readline()

        # 读取点云数据部分
        points = []
        for line in file:
            # 假设每行数据为：x y z r g b
            coords = list(map(float, line.strip().split()))
            if len(coords) == 6:  # 包含 x, y, z, r, g, b
                points.append(coords)
        
        # 转换为numpy数组，列分别是 x, y, z, r, g, b
        points = np.array(points)
        return points
    
def calculate_point_cloud_error(cloud1, cloud2):
    """计算两个点云之间的均方误差(MSE)"""
    if cloud1.shape != cloud2.shape:
        raise ValueError("两个点云的点数必须相同")
    
    # 计算每对对应点的欧氏距离
    squared_diff = np.sum((cloud1 - cloud2) ** 2, axis=1)  # 每个点的平方差
    mse = np.mean(squared_diff)  # 平均均方误差
    return mse

def calculate_absolute_error(cloud1, cloud2):
    """计算两个点云之间的绝对误差"""
    if cloud1.shape != cloud2.shape:
        raise ValueError("两个点云的点数必须相同")
    
    # 计算每对对应点的坐标差的绝对值
    absolute_diff = np.abs(cloud1[:, :3] - cloud2[:, :3])  # 只计算x, y, z坐标的绝对差异
    
    # 计算每个点的总绝对误差（x, y, z 分别的误差）
    total_absolute_error = np.sum(absolute_diff, axis=1)  # 对每个点的绝对误差求和
    mean_absolute_error = np.mean(total_absolute_error)  # 平均绝对误差
    
    return mean_absolute_error

# 示例使用
file_path1 = 'output_Curasao.ply'  # 替换为第一个点云文件路径curasao
file_path2 = 'curasao.ply'  # 替换为第二个点云文件路径

# 读取两个点云
point_cloud1 = read_ply_points(file_path1)
point_cloud2 = read_colmap_ply_with_plyfile(file_path2)

# 计算误差
error = calculate_point_cloud_error(point_cloud1, point_cloud2)
print(f"两个点云之间的均方误差 (MSE) 为: {error}")

# 计算绝对误差
error = calculate_absolute_error(point_cloud1, point_cloud2)
print(f"两个点云之间的平均绝对误差 (Mean Absolute Error) 为: {error}")
