clc;
close all;
clear;

% ==================== 参数设置 ====================
obj_file = "expanded_ball_surface.obj";       % OBJ模型路径
output_folder = "..\..\png\expanded_ball_surface_ori";  % 输出文件夹（需提前创建）
views = {                    % 定义视角列表 [az, el] + 视图名称（用于文件名）
    [0, 90],   'Top_View';     
    [0, 0],    'Front_View';   
    [90, 0],   'Side_View';    
    [45, 30],  'Isometric_View' 
};
fig_size = [1600, 1200];     % 图片尺寸（宽×高，像素）

% ==================== 主流程 ====================
% 创建并设置figure
fig = figure('Position', [100, 100, fig_size(1), fig_size(2)], 'Visible', 'off'); % 隐藏窗口（可选）
hold on;
axis equal;
grid on;
xlabel('X'); ylabel('Y'); zlabel('Z');

% 读取模型数据
[V, F] = readObjMesh(obj_file);

% 循环处理每个视角
for i = 1:length(views)
    % 清除当前视图（保留坐标轴设置）
    cla;
    
    % 绘制模型（红色边缘，无填充）
    patch('Faces', F, 'Vertices', V, ...
          'FaceColor', 'none', 'EdgeColor', 'red', 'LineWidth', 1);
    
    % 设置当前视角
    view(views{i,1}(1), views{i,1}(2));  % [az, el]
    
    % 可选：调整坐标轴范围（根据模型实际大小自动适配）
    axis tight;
    
    % 可选：隐藏刻度（若需要更简洁的图片）
    set(gca, 'XTick', [], 'YTick', [], 'ZTick', []);
    
    % 构造输出文件名（含视角名称）
    filename = sprintf('%s(%s).png', output_folder, views{i,2});
    
    % 保存图片（高DPI）
    saveas(gcf, filename);
end

disp('所有视图已保存完毕！');

% ==================== 辅助函数（读取OBJ） ====================
function [V, F] = readObjMesh(filename)
    fid = fopen(filename, 'r');
    if fid == -1
        error('无法打开文件：%s', filename);
    end
    V = [];
    F = [];
    while ~feof(fid)
        line = strtrim(fgetl(fid));
        if isempty(line)
            continue;
        end
        if startsWith(line, 'v ')
            % 顶点数据：v x y z
            nums = sscanf(line(2:end), '%f %f %f');
            V(end+1, :) = nums'; %#ok<AGROW>
        elseif startsWith(line, 'f ')
            % 面数据：f v1 v2 v3（或v1/vt1/vn1...）
            idx = regexp(line(2:end), '\d+', 'match');
            idx = cellfun(@str2double, idx);
            if numel(idx) >= 3
                % 三角形面（直接取前3个顶点）
                F(end+1, :) = idx(1:3)'; %#ok<AGROW>
                % 四边形面（拆分为两个三角形，可选）
                % if numel(idx) == 4
                %     F(end+1, :) = idx([1,2,3])';
                %     F(end+1, :) = idx([1,3,4])';
                % end
            end
        end
    end
    fclose(fid);
    % 转换为整数索引（OBJ索引从1开始，Matlab兼容）
    F = double(F);
end