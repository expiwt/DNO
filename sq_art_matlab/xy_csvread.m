function xy_data_cell = xy_csvread(csv_filename)
    data = readmatrix(csv_filename);
    xy_data_cell = cell(size(data, 1), 1);
    
    for i = 1:size(data, 1)
        xy_data_cell{i} = data(i,:);
    end
end