function [results, c_func] = solve_pde(partsize, cof, scaled, scale_c)
    % Решение PDE для геометрических данных
    
    model = createpde;
    
    % Создание геометрии (четырехугольник)
    P1 = [2, 4, partsize(1), partsize(3), partsize(6), partsize(5), ...
          partsize(2), partsize(4), 0, 0]';
    
    gm = P1;
    sf = 'P1';
    ns = char('P1')';
    
    g = decsg(gm, sf, ns);
    geometryFromEdges(model, g);
    
    % Определение коэффициента
    c_func = @(region, state) (cof(1) * sin(pi * (region.x / scaled / 10)) - ...
                              cof(2) .* (region.x / scaled) .* (region.x / scaled - 10) + 2) * scale_c;
    
    % Граничные условия
    applyBoundaryCondition(model, 'dirichlet', 'Edge', 1:model.Geometry.NumEdges, 'u', 0);
    
    % Коэффициенты PDE
    specifyCoefficients(model, 'm', 0, 'd', 0, 'c', c_func, 'a', 0, 'f', 1);
    
    % Создание сетки
    generateMesh(model, 'Hmax', 0.25 * scaled);
    
    % Решение
    results = solvepde(model);
end

