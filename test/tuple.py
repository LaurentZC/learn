# 创建一个元组
my_tuple = (1, 2, 2, 3, 4, 5)

# 计算元组长度
length = len(my_tuple)
print(f"Length: {length}")

# 查找元素的位置
index = my_tuple.index(3)  # 查找元素3的位置
print(f"Index of 3: {index}")

# 计数某个元素出现的次数
count = my_tuple.count(2)
print(f"Count of 2: {count}")

# 访问元组元素
element = my_tuple[2]  # 访问第三个元素
print(f"Element at index 2: {element}")

# 切片操作
slice_tuple = my_tuple[1:4]
print(f"Slice from index 1 to 3: {slice_tuple}")

# 拼接元组
new_tuple = my_tuple + (6, 7)
print(f"Concatenated tuple: {new_tuple}")

# 复制元组
copied_tuple = my_tuple * 2
print(f"Copied tuple: {copied_tuple}")
