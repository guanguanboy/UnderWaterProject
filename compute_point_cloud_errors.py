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
