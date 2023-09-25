%% Initialisation
clear;
close all;
clc;

%% Map initialization
H = 10;
W = 15;
res = 0.2;
[X, Y] = meshgrid([-W/2:res:W/2], [-H/2:res:H/2]);
map = zeros(H/0.2+1, W/0.2+1);

% Obstacles [x1, y1, x2, y2]'
obstacles = [[-W/3; H/2; -W/4; -H/5], [-2; H/2.8; 0; -H/2], [W/10; H/2; W/2.8; 0]];
% Start / End
x0 = [-6; -4];
xd = [6; -1];

% Plot Map
figure(1);
rectangle('Position', [-W/2, -H/2, W, H], 'FaceColor', 'white');
for k = 1:size(obstacles, 2)
    obsk = obstacles(:, k);
    obsk = [obsk(1), obsk(4), obsk(3)-obsk(1), obsk(2)-obsk(4)];
    hold on;
    rectangle('Position', obsk, 'FaceColor', 'blue');
end
hold on;
plot([x0(1), xd(1)], [x0(2), xd(2)], 'rx', 'LineWidth', 2);

xlim([-W/2-1 W/2+1]);
ylim([-H/2-1 H/2+1]);
grid on;

%% RAPIDLY EXPLORING RANDOM TREES (RRT)
% Initialize tree, step size, distribution, etc..
Nmax = 500; % Max number of points
K = 5; % Number of neirest neighboors
l = 0.8;
lmax = 1.;
use_rrt_star = true;
dt = 0.1;
tree = {};
tree{1} = {x0, 0, 0}; % {coordinates, parent index, cost}
hold on; 
p = plot(0, 0);
disp("Press any key to start RRT..");
pause;

tic
for i=2:1:Nmax
    collision=true;
    while(collision)
        % Sample randomly (uniform) one point in the map
        Xi = (rand([2, 1])-0.5) .* [W; H];

        % Take the closest point from the tree
        dist_min = 1e6;
        idx_min = -1;
        for j = 1:i-1
            dist = norm(tree{j}{1} - Xi, 2);
            if dist < dist_min
                dist_min = dist;
                idx_min = j;
            end
        end

        costi = norm(Xi - tree{idx_min}{1}, 2);
        u = (Xi - tree{idx_min}{1})/costi;
        xi = tree{idx_min}{1} + l*u;
        
        % Check if the new point is feasible (no collisions)
        collision = false;
        if xi(1) >= W/2 || xi(1) <= -W/2 || xi(2) >= H/2 || xi(2) <= -H/2 % Check that points remain within the arena
            collision=true;
        else
            for k=1:size(obstacles, 2)
                obsk = obstacles(:, k);
                if xi(1) <= obsk(3) && xi(1) >= obsk(1) && xi(2) <= obsk(2) && xi(2) >= obsk(4) % Check for obstacle collision
                    collision=true;
                end
            end
        end
    end
    
    % Check neighborhood to see if the cost is smaller
    if i>2 && use_rrt_star% Without that condition, we only have the starting point in the tree
        [DNN, INN, XNN] = knn(xi, tree, K);
        min_cost = tree{idx_min}{3} + costi;
        for k = 1:size(XNN, 2) % Go through the kNN and see if the total cost is smaller than with initial parent
            costk = tree{INN(k)}{3} + DNN(k);
            if costk < min_cost && DNN(k) < lmax
                min_cost = costk;
                idx_min = INN(k);
                costi = DNN(k);
            end
        end
        
        hold on;
        delete(p)
        p = plot(XNN(1, :), XNN(2, :), 'o', 'Color', 'red');
    end
    
    
    % Add it to the tree
    tree{i} = {xi, idx_min, costi + tree{idx_min}{3}};
    % Plot new link
    xparent = tree{idx_min}{1};
    hold on;
    plot(xi(1), xi(2), 'o', 'LineWidth', 2, 'Color', 'green');
    hold on;
    plot([xi(1), xparent(1)], [xi(2), xparent(2)], 'Color', 'green');
    title(strcat(" Iter ", num2str(i), "/ ", num2str(Nmax)));
    
    
    % Stop if the point is within the tolerance
    if norm(xi - xd, 2) < 1
        break;
    end
    
    pause(dt);
end
toc

%% Plot final trajectory
trajectory = [];
idx = i;
while idx ~= 0
    trajectory = [tree{idx}{1} trajectory];
    idx = tree{idx}{2};
end

hold on;
plot(trajectory(1, :), trajectory(2, :), 'Color', 'red', 'LineWidth', 3);
title("Final Cost : ", num2str(tree{i}{3}));
%% FUNCTION
function [D, I, X] = knn(x, tree, K)
    dist = [];
    idx = [];
    for i = 1:length(tree)
        [dist, I] = sort([dist, norm(tree{i}{1}-x, 2)]);
        idx = [idx, i];
        idx = idx(I);        
    end
    if K > i
        K = i-1;
    end
    X = [];
    D = [];
    for k=1:K
        X = [tree{idx(k)}{1} X];
        D = [dist(idx(k)) D];
    end
    I = idx;
end



