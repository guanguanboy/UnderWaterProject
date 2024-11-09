import pycolmap
import numpy as np

def points3D_to_ply(colmap_project_path, output_ply_file):
        # 加载重建模型
    reconstruction = pycolmap.Reconstruction(colmap_project_path)

    # 访问三维点信息
    #for point3D_id, point3D in reconstruction.points3D.items():
    #    print(f"3D Point ID: {point3D_id}")
    #    print(f"Coordinates: {point3D.xyz}")
    
    # 写PLY文件的头部
    with open(output_ply_file, 'w') as ply_file:
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write("element vertex {}\n".format(reconstruction.num_points3D()))  # 点的数量
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("end_header\n")

        # 写入点的坐标
        for point_id, point in reconstruction.points3D.items():
            ply_file.write("{:.6f} {:.6f} {:.6f}\n".format(point.xyz[0], point.xyz[1], point.xyz[2]))


#def convert_points3D_to_ply(colmap_project_path, output_ply_file):
    
    ## 加载 points3D 数据
    #points3D = pycolmap.Point3D(colmap_project_path + '/points3D.bin').todict
    
    # 将 points3D 转换为 PLY 文件
#    points3D_to_ply(points3D, output_ply_file)
#    print(f"PLY file saved to {output_ply_file}")


# 使用示例
colmap_project_path = 'SeathruNeRF_dataset/Panama/sparse/0/'  # COLMAP 项目路径
output_ply_file = 'output_Panama.ply'  # 输出的 PLY 文件路径


points3D_to_ply(colmap_project_path, output_ply_file)
