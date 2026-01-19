import numpy as np

# Количество геометрий
num = 30

# Их номера
part_num = np.arange(1, num + 1).reshape(-1, 1)

# Генерация параметров для четырехугольника
leftpoint_x = np.random.randint(0, 21, num) / 10.0      # 0.0 - 2.0
leftpoint_y = np.random.randint(40, 61, num) / 10.0     # 4.0 - 6.0

rightpoint_x = np.random.randint(80, 101, num) / 10.0   # 8.0 - 10.0  
rightpoint_y = np.random.randint(40, 61, num) / 10.0    # 4.0 - 6.0

down_leftpoint_x = np.random.randint(20, 41, num) / 10.0  # 2.0 - 4.0

down_rightpoint_x = np.random.randint(60, 81, num) / 10.0 # 6.0 - 8.0

leftpoint = np.column_stack([leftpoint_x, leftpoint_y])
rightpoint = np.column_stack([rightpoint_x, rightpoint_y])
down_leftpoint = np.column_stack([down_leftpoint_x])
down_rightpoint = np.column_stack([down_rightpoint_x])


# Формируем part_size как конкатенацию всех массивов по горизонтали
part_size = np.column_stack([
    leftpoint,      # левая верхняя [x1, y1]
    rightpoint,     # правая верхняя [x2, y2]  
    down_leftpoint, # левая нижняя [x3]
    down_rightpoint # правая нижняя [x4]
])

print("part_size shape:", part_size.shape) 
print(part_size[1])


# Генерация случайных масштабов
random_numbers = np.random.rand(num, 1)
random_numbers = 0.5 + 0.5 * random_numbers
scaled = np.round(random_numbers, 1)

print("random_numbers shape:", random_numbers.shape)
print("scaled shape:", scaled.shape)
print("Пример scaled:", scaled[:5].flatten())

data = np.column_stack([part_num, part_size * scaled, scaled])

np.savetxt('./sq_art/geo_data_quad.csv', data, delimiter=',', fmt='%.2f')

#формат geo_data_quad.csv:
#индекс точки, x_left_up_scaled, y_left_up_scale, x_right_up_scaled, y_right_up_scale, x_left_down_scaled, x_right_down_scale, scaled