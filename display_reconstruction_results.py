import pycolmap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置COLMAP项目的路径
colmap_project_path = 'SeathruNeRF_dataset/Curasao/sparse/0/'

# 加载重建结果
cameras = pycolmap.read_cameras_binary(colmap_project_path + '/cameras.bin')
images = pycolmap.read_images_binary(colmap_project_path + '/images.bin')
points3D = pycolmap.read_points3D_binary(colmap_project_path + '/points3D.bin')

# 可视化点云
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 从points3D中提取坐标
xs = []
ys = []
zs = []
for point_id, point in points3D.items():
    xs.append(point.xyz[0])
    ys.append(point.xyz[1])
    zs.append(point.xyz[2])

# 绘制点云
ax.scatter(xs, ys, zs, c='r', marker='o')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('3D Point Cloud Visualization')
plt.show()

# 可视化相机位置
for image_id, image in images.items():
    cam_center = image.camera.center
    ax.scatter(cam_center[0], cam_center[1], cam_center[2], c='b', marker='^', s=100)

plt.show()
