function data_generate()
% DATA_GENERATE - основной скрипт генерации данных

clear;

fprintf('Начинаем генерацию данных...\n');

% Загружаем геометрические данные
fprintf('Загрузка геометрических данных...\n');
part_size_file_ = load_geo_data('./geo_new.csv');
n_obj = length(part_size_file_);

fprintf('Загружено %d объектов\n', n_obj);

% Загружаем координатные данные
fprintf('Загрузка координатных данных...\n');
X_ = xy_csvread('./data/x_data.csv');
Y_ = xy_csvread('./data/y_data.csv');

% Проверяем соответствие размеров
if length(X_) ~= n_obj || length(Y_) ~= n_obj
    error('Размеры X_data и Y_data не соответствуют количеству объектов');
end

% Инициализируем пустые массивы для всех объектов
U_all = [];
C_all = [];

fprintf('Начинаем обработку объектов...\n');

for obj_idx = 1:n_obj
    fprintf('Обработка объекта %d из %d...\n', obj_idx, n_obj);
    
    % Данные текущего объекта
    current_geo = part_size_file_{obj_idx};
    
    num = 1; % Один объект = одна геометрия
    all_cof = randi([20,80], num, 2);
    all_cof = all_cof./100;

    X = X_{obj_idx};
    Y = Y_{obj_idx};
    
    part_num = current_geo.index;
    scaled = current_geo.scaled;
    part_size = [current_geo.x_left_up, current_geo.y_left_up, ...
                 current_geo.x_right_up, current_geo.y_right_up, ...
                 current_geo.x_left_down, current_geo.x_right_down];
    
    % Инициализируем для текущего объекта
    U_obj = [];
    C_obj = [];

    for part_idx = 1:num
        size_coords = part_size;
        cof = all_cof(part_idx, :);
        x = X;
        y = Y;
        
        % Вычисляем масштаб C
        if rand > 0.5
            scale_c = 0.01 + 0.02 * randi(50);
        else
            scale_c = 1 + 0.05 * randi(60);
        end

        % Решение PDE на физ области 
        fprintf('  Решение PDE...\n');
        [results, cc] = solve_pde(size_coords, cof, scaled, scale_c);
        
        % Коорд. узлов треуг сетки в физ области
        xy = results.Mesh.Nodes;

        % Решение PDE в этих узлах
        ui = results.NodalSolution;

        % Интерполяция на регулярную сетку
        fprintf('  Интерполяция на регулярную сетку...\n');
        u = griddata(xy(1,:), xy(2,:), ui, x, y, 'cubic');
        
        % Убедимся, что u - вектор-строка для конкатенации
        if size(u, 1) > size(u, 2)
            u = u';
        end
        
        U_obj = [U_obj; u];
        
        % Вычисляем C
        c = cof(1) * sin(pi * (x / scaled / 10)) - cof(2) .* (x / scaled) .* (x / scaled - 10) + 2;
        
        if size(c, 1) > size(c, 2)
            c = c';
        end
        
        C_obj = [C_obj; c * scale_c];
    end
    
    % Добавляем результаты текущего объекта к общим
    U_all = [U_all; U_obj];
    C_all = [C_all; C_obj];
    
    fprintf('  Объект %d завершен\n', obj_idx);
end

% Сохраняем объединенные результаты
fprintf('Сохранение результатов...\n');
csvwrite('./data/U.csv', U_all);
csvwrite('./data/C.csv', C_all);

fprintf('Генерация данных завершена!\n');
fprintf('Размер U: %s\n', mat2str(size(U_all)));
fprintf('Размер C: %s\n', mat2str(size(C_all)));

end