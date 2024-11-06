def read_ply_data(file_path):
    with open(file_path, 'r') as file:
        # 读取文件头部
        header = []
        line = file.readline()
        while line.strip() != 'end_header':
            header.append(line.strip())
            line = file.readline()
        
        # 读取点云数据部分
        data = []
        for line in file:
            data.append(line.strip())

        return data

# 使用示例
file_path = 'output_Curasao.ply'  # 替换为实际的PLY文件路径
point_cloud_data = read_ply_data(file_path)

# 输出点云数据
for line in point_cloud_data[:10]:  # 只打印前10个点的数据，可以根据需要调整
    print(line)
