import pickle

# 读取pkl文件
with open('./metric_dict.pkl', 'rb') as f:  # 注意是 'rb' 二进制读取模式
    data = pickle.load(f)

# 查看数据内容
print(data)
print(type(data))