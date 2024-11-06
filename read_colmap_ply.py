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

# 使用示例
file_path = 'curasao.ply'  # 替换为COLMAP导出的PLY文件路径
point_cloud = read_colmap_ply_with_plyfile(file_path)

# 输出点云的前10个点
print("点云数据（前10个点）：")
print(point_cloud[:10])
