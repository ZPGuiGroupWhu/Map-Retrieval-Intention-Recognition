import json

# 读取原始文件
with open("Lall_ancestors.json", "r") as f:
    data = json.load(f)

# 提取所有的 key
keys = list(data.keys())

root = {}
for i in keys:
    if i == "Global":
        root[i] = []
    else:
        root[i] = ["Global"]

# 将 keys 存储到新的 JSON 文件中
with open("Lall_concepts.json", "w") as f:
    json.dump(keys, f)

# 将 keys 存储到新的 JSON 文件中
with open("Lall_roots.json", "w") as f:
    json.dump(root, f)