from transformers import AutoTokenizer
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

tokenizer = AutoTokenizer.from_pretrained("/data03/sunyi/hf_cache/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/14dd1130311655b43c3ce41dd505f70f6ca89845")

dataset_path = "/path/to/you/datasets/DATA.jsonl"
output_file_path = "/path/to/you/datasets/DATA_10k.jsonl"
token_counts = []
filtered_data = []
with open(dataset_path, "r", encoding="utf-8") as files:
    for file in tqdm(files):
        data = json.loads(file)  

        # for item in tqdm(data):
        combined_text = data["prompt"] + " " + data["response"]

        tokens = tokenizer.tokenize(combined_text)

        token_counts.append(len(tokens))
        # 如果token数量不超过10k，则存储该条目
        if len(tokens) <= 10000:
            filtered_data.append(data)
            # 也可以直接写入文件，而不是先存储再写入
            output_file.write(json.dumps(data) + '\n')

# 将过滤后的数据写入新的JSONL文件
with open(output_file_path, "w", encoding="utf-8") as output_file:
    for data in filtered_data:
        output_file.write(json.dumps(data) + '\n')

print(f"Total lines: {len(token_counts)}")
print(f"Max tokens per line: {max(token_counts)}")
print(f"Min tokens per line: {min(token_counts)}")
print(f"Average tokens per line: {sum(token_counts) / len(token_counts)}")


plt.figure(figsize=(10, 6))
sns.histplot(token_counts, bins=50, kde=True)
plt.title("Distribution of Token Counts per Line")
plt.xlabel("Token Count")
plt.ylabel("Frequency")
plt.grid(True)

plt.savefig("token_count_histogram.png")
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x=token_counts)
plt.title("Boxplot of Token Counts per Line")
plt.xlabel("Token Count")
plt.grid(True)

plt.savefig(f"token_count_boxplot.png")
plt.show()