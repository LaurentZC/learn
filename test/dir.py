# 创建一个字典
my_dict = {'name': 'Charon', 'age': 20, 'city': 'Qing Dao', 'email': 'hello@gmail.com', 'null': 1}

# 打印一下
print("Original Dictionary: ", my_dict)

# 删除一个键值对
del my_dict['null']
# 弹出
city = my_dict.pop('city')
print("Dictionary after delete: ", my_dict)
print("city: ", city)

# 获取一个值，使用 get 方法，如果键不存在则返回默认值
email = my_dict.get('email', 'Not Found')
print("email: ", email)

# 添加一个新的键值对
my_dict['sex'] = 'male'
print("Dictionary after update: ", my_dict)

# 获取所有键
keys = my_dict.keys()
print("Keys:", keys)

# 获取所有值
values = my_dict.values()
print("Values:", values)

# 获取所有键值对
items = my_dict.items()
print("Items:", items)

# 清空字典
my_dict.clear()

# 输出结果
print("Dictionary after clear:", my_dict)
