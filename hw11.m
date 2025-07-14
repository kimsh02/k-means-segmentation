function [q] = hw11()
% usage:
%  q = hw11()
% where q is the number of questions answered
% 
%  assumes all required files for hw11 exist in the same directory as the
%  script

format compact
close all
q = 3;

% A

% 1

sunset = imread("sunset.tiff");
tiger1 = imread("tiger-1.tiff");
tiger2 = imread("tiger-2.tiff");

sunset = double(sunset);
tiger1 = double(tiger1);
tiger2 = double(tiger2);

function [centroids, labels] = kmeans_clustering(X, K, max_iters)
    centroids = X(randperm(size(X, 1), K), :);
    labels = zeros(size(X, 1), 1);
    for iter = 1:max_iters
        for i = 1:size(X, 1)
            distances = sum((centroids - X(i, :)).^2, 2);
            [~, labels(i)] = min(distances);
        end
        new_centroids = zeros(K, size(X, 2));
        for k = 1:K
            cluster_points = X(labels == k, :);
            if ~isempty(cluster_points)
                new_centroids(k, :) = mean(cluster_points);
            end
        end
        objective = 0;
        for i = 1:size(X, 1)
            objective = objective + sum((X(i, :) - centroids(labels(i), :)).^2);
        end
        disp(['Iteration ', num2str(iter), ', Objective function: ', num2str(objective)]);
        if all(new_centroids == centroids)
            break;
        end
        centroids = new_centroids;
    end
end

K5 = 5;
K10 = 10;
max_iters = 200;
[m, n, ~] = size(sunset);
s = reshape(sunset, m*n, 3);
[m, n, ~] = size(tiger1);
t1 = reshape(tiger1, m*n, 3);
[m, n, ~] = size(tiger2);
t2 = reshape(tiger2, m*n, 3);
imgs = {s, t1, t2};

if false
for idx = 1:size(imgs,2)
    [c5, l5] = kmeans_clustering(imgs{idx}, K5, max_iters);
    segmented_img5 = reshape(c5(l5,:), m, n, 3);
    [c10, l10] = kmeans_clustering(imgs{idx}, K10, max_iters);
    segmented_img10 = reshape(c10(l10,:), m, n, 3);
    figure;
    imshow(uint8(segmented_img5));
    % title('Segmentation with K=5');
    set(gca,'Position', [0.05 0.05 0.9 0.9]);
    saveas(gcf, sprintf('%da5.png', idx));
    figure;
    imshow(uint8(segmented_img10));
    % title('Segmentation with K=10');
    set(gca,'Position', [0.05 0.05 0.9 0.9]);
    saveas(gcf, sprintf('%da10.png', idx));
end
end

inputs = {sunset, sunset, tiger1, tiger1, tiger2, tiger2};
imgs = {};

for idx = 1:size(inputs,2)
img = inputs{idx};
[rows, cols, ~] = size(img);
output = zeros(rows * cols, 5);
index = 1;
for y = 1:cols
    for x = 1:rows
        r = img(x, y, 1);  % Red channel
        g = img(x, y, 2);  % Green channel
        b = img(x, y, 3);  % Blue channel
        if mod(idx, 2) == 1
        output(index, :) = [r, g, b, (x-1)/(rows-1)*255, (y-1)/(cols-1)*255];
        else
        output(index, :) = [r, g, b, (x-1)/(rows-1)*255*10, (y-1)/(cols-1)*255*10];
        end
        index = index + 1;
    end
end
imgs{end+1} = output;
end

ctr = 1;
if false
for idx = 1:size(imgs,2)
    [c10, l10] = kmeans_clustering(imgs{idx}, K10, max_iters);
    segmented_img10 = reshape(c10(l10,:), m, n, 5);
    figure;
    imshow(uint8(segmented_img10(:,:,1:3)));
    % title('Segmentation with K=10');
    set(gca,'Position', [0.05 0.05 0.9 0.9]);
    if mod(idx, 2) == 1
        saveas(gcf, sprintf('%db1.png', ctr));
    else
        saveas(gcf, sprintf('%db10.png', ctr));
        ctr = ctr + 1;
    end
end
end

dx = [1 -1];
dy = [1; -1];
sunset = double(rgb2gray(imread('sunset.tiff')));
tiger1 = double(rgb2gray(imread('tiger-1.tiff')));
tiger2 = double(rgb2gray(imread('tiger-2.tiff')));

sigma = 4;
dim = 6 * sigma + 1;
A = zeros(dim,dim);
for x=1:dim
    for j=1:dim
        A(x,j)=exp((-(x-ceil(dim/2))^2-(j-ceil(dim/2))^2)/(2*sigma^2));
    end
end
A = A/sum(A(:));

sigma = 2;
dim = 6 * sigma + 1;
B = zeros(dim,dim);
for x=1:dim
    for j=1:dim
        B(x,j)=exp((-(x-ceil(dim/2))^2-(j-ceil(dim/2))^2)/(2*sigma^2));
    end
end
B = B/sum(B(:));

grays = {sunset, tiger1, tiger2};
imgs4d = {};
windowSize = [31, 31];
kernel = ones(windowSize)/ (windowSize(1) * windowSize(2));
sunset_c = imread("sunset.tiff");
tiger1_c = imread("tiger-1.tiff");
tiger2_c = imread("tiger-2.tiff");
sunset_c = double(sunset_c);
tiger1_c = double(tiger1_c);
tiger2_c = double(tiger2_c);
color_imgs = {sunset_c, tiger1_c, tiger2_c};
filtxA = conv2(A, dx, 'same');
filtyA = conv2(A, dy, 'same');
filtxB = conv2(B, dx, 'same');
filtyB = conv2(B, dy, 'same');
lambda = 1;

for idx = 1:size(grays,2)
[m, n] = size(grays{idx});
output = zeros(m,n,4);
eb4_x = conv2(grays{idx}, filtxA, 'same');
eb4_y = conv2(grays{idx}, filtyA, 'same');
eb2_x = conv2(grays{idx}, filtxB, 'same');
eb2_y = conv2(grays{idx}, filtyB, 'same');
% gmag = sqrt(eb4_x.^2 + eb4_y.^2);
% for id = 1:size(grad,1)
%     for j = 1:size(grad,2)
%         if gmag(id,j) > th
%             grad(id,j) = 255;
%         else
%             grad(id,j) = 0;
%         end
%     end
% end
% figure;
% hold on;
% imshow(gmag);
% axis image;
% set(gca,'Position', [0.05 0.05 0.9 0.9]);
M = zeros(m, n, 2);
[M(:,:,1), M(:,:,2)] = ndgrid(1:m, 1:n);
ebs = {eb4_x,eb4_y,eb2_x,eb2_y, color_imgs{idx}(:,:,1), color_imgs{idx}(:,:,2), color_imgs{idx}(:,:,3), M(:,:,1),M(:,:,2)};
for f = 1:size(ebs,2)
    if f <= 4
        convd = sqrt(conv2(ebs{f}.^2, kernel, 'same'));
        output(:,:,f) = convd ./ max(convd(:)) * 255;
        % output(:,:,f) = ebs{f};
    end
    if f > 4 && f <= 7
        output(:,:,f) = ebs{f};
    end
    if f == 8
        output(:,:,f) = ((ebs{f}-1)./(m-1)).*255*lambda;
    end
    if f == 9
        output(:,:,f) = ((ebs{f}-1)./(n-1)).*255*lambda;
    end
end
imgs4d{end+1} = output;
end

imgs = {};
for a = 1:size(imgs4d,2)
    [m,n,o] = size(imgs4d{a});
    imgs{end+1} = reshape(imgs4d{a}, m*n, 9);
end

function [centroids, labels] = kmeans_clustering_select(X, K, max_iters, select)
    centroids = X(randperm(size(X, 1), K), :);
    labels = zeros(size(X, 1), 1);
    for iter = 1:max_iters
        for i = 1:size(X, 1)
            distances = sum((centroids(:,1:select) - X(i, 1:select)).^2, 2);
            [~, labels(i)] = min(distances);
        end
        new_centroids = zeros(K, size(X, 2));
        for k = 1:K
            cluster_points = X(labels == k, :);
            if ~isempty(cluster_points)
                new_centroids(k, :) = mean(cluster_points);
            end
        end
        objective = 0;
        for i = 1:size(X, 1)
            objective = objective + sum((X(i, 1:select) - centroids(labels(i), 1:select)).^2);
        end
        disp(['Iteration ', num2str(iter), ', Objective function: ', num2str(objective)]);
        if all(new_centroids == centroids)
            break;
        end
        centroids = new_centroids;
    end
end

if true
for idx = 1:size(imgs,2)
    for part = 1:3
        if mod(part, 3) == 1
            [c10, l10] = kmeans_clustering_select((imgs{idx}), K10, max_iters, 4);
            segmented_img10 = reshape(c10(l10,:), m, n, 9);
            fig = figure;
            imshow(uint8(segmented_img10(:,:,5:7)));
            % title('Segmentation with K=10');
            set(gca,'Position', [0.05 0.05 0.9 0.9]);
            saveas(fig, sprintf('p%dimg%dc.png', part, idx));
            continue;
        end
        if mod(part, 3) == 2
            [c10, l10] = kmeans_clustering_select((imgs{idx}), K10, max_iters, 7);
            segmented_img10 = reshape(c10(l10,:), m, n, 9);
            fig = figure;
            imshow(uint8(segmented_img10(:,:,5:7)));
            % title('Segmentation with K=10');
            set(gca,'Position', [0.05 0.05 0.9 0.9]);
            saveas(fig, sprintf('p%dimg%dc.png', part, idx));
            continue;
        end
        if mod(part, 3) == 0
            [c10, l10] = kmeans_clustering_select((imgs{idx}), K10, max_iters, 9);
            segmented_img10 = reshape(c10(l10,:), m, n, 9);
            fig = figure;
            imshow(uint8(segmented_img10(:,:,5:7)));
            % title('Segmentation with K=10');
            set(gca,'Position', [0.05 0.05 0.9 0.9]);
            saveas(fig, sprintf('p%dimg%dc.png', part, idx));
            continue;
        end
    end
end
end
end