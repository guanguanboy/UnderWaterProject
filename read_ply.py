def read_ply_header(file_path):
    with open(file_path, 'r') as file:
        # 读取文件头部
        header = []
        line = file.readline()
        while line.strip() != 'end_header':
            header.append(line.strip())
            line = file.readline()

        return header

# 使用示例
file_path = 'output_Curasao.ply'  # 替换为实际的PLY文件路径
header_info = read_ply_header(file_path)

# 输出头部信息
for line in header_info:
    print(line)
