import json
import os


output_file_path = '/data/user_name/Projects/datasets/unseen-taxonomic_cleaned_800_cate_.json'  # 输出 JSON 文件路径
replacement_file_path = '/data/user_name/Projects/datasets/examples_of_800unseen.json'
input_file_path = '/data/user_name/Projects/datasets/unseen-taxonomic_cleaned.json'

key_to_replace = 'categories'
replacement_key = 'categories'

# 读取第一个 JSON 文件
with open(input_file_path, 'r') as f:
    data = json.load(f)

# 读取第二个 JSON 文件
with open(replacement_file_path, 'r') as f:
    replacement_data = json.load(f)

# 检查键是否存在
if key_to_replace in data and replacement_key in replacement_data:
    # 替换值
    data[key_to_replace] = replacement_data[replacement_key]
else:
    print("键不存在，请检查输入的键是否正确。")

# 将修改后的数据写入输出文件
with open(output_file_path, 'w') as f:
    json.dump(data, f, indent=4)