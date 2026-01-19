function geo_data_cell = load_geo_data(csv_filename)
    % Загружает геометрические данные из CSV файла
    
    data = readmatrix(csv_filename);
    
    geo_data_cell = cell(size(data, 1), 1);
    
    for i = 1:size(data, 1)
        geo_data = struct();
        geo_data.index = data(i, 1);
        geo_data.x_left_up = data(i, 2);
        geo_data.y_left_up = data(i, 3);
        geo_data.x_right_up = data(i, 4);
        geo_data.y_right_up = data(i, 5);
        geo_data.x_left_down = data(i, 6);
        geo_data.x_right_down = data(i, 7);
        geo_data.scaled = data(i, 8);
        
        geo_data_cell{i} = geo_data;
    end
end