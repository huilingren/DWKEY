# 使用最大值最小值方法将权重转为0-1区间
def normalize_gml_weights(gml_input_path, gml_output_path):
    # 第一次遍历：提取所有 weight 值
    weights = []
    with open(gml_input_path, 'r') as fin:
        for line in fin:
            line = line.strip()
            if line.startswith('weight'):
                try:
                    weight_val = float(line.split()[1])
                    weights.append(weight_val)
                except:
                    continue

    if not weights:
        print("未找到任何权重字段。请确认 GML 文件中包含 edge 的 weight 字段。")
        return

    w_min = min(weights)
    w_max = max(weights)
    print(f"权重范围: min = {w_min}, max = {w_max}")

    # 第二次遍历：写入新文件，归一化 weight
    with open(gml_input_path, 'r') as fin, open(gml_output_path, 'w') as fout:
        for line in fin:
            if line.strip().startswith('weight'):
                try:
                    original_weight = float(line.strip().split()[1])
                    normalized = (original_weight - w_min) / (w_max - w_min)
                    fout.write(f"    weight {normalized:.6f}\n")
                except:
                    fout.write(line)
            else:
                fout.write(line)

    print(f"已完成归一化，结果写入：{gml_output_path}")


# 示例用法
if __name__ == "__main__":
    input_path = "/home/renhuiling/Code_Ren/data/real/Cost/coin_degree.gml"      # 你的原始 GML 文件路径
    output_path = "coin_degree_normalnized.gml"  # 归一化后的输出路径
    normalize_gml_weights(input_path, output_path)