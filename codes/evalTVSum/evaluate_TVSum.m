function [stats] = evaluate_TVSum(videos_test, teInds)
addpath('./knapsack');
load ./tvsum50.mat; % load the dataset

n_videos = numel(tvsum50);
budget = 0.18;
%% Normalize data
for i=1:length(videos_test)
    videos_test(i).Ypred = videos_test(i).Ypred - min(videos_test(i).Ypred);
    videos_test(i).frame_score = videos_test(i).Ypred / max(videos_test(i).Ypred);
end
%% Generate segmentation results
load(['../../data/shot_TVSum.mat']);
seg_idx = cell(1,n_videos);
for i=1:n_videos,
    shot_boundary = shot_boundaries{i};
    seg_tmp = zeros(numel(shot_boundary)-1,2);
    for j=1:numel(shot_boundary)-1,
        seg_tmp(j,2) = shot_boundary(j+1);
        seg_tmp(j,1) = shot_boundary(j)+1;
    end
    seg_idx{i} = seg_tmp;
end


%% Evaluate performance
stats = struct;
% 50 videos
for i = teInds
%     disp(i)
    % Get prediction label
    pred_lbl = videos_test(i).frame_score;
    
    % Get predicted shot segments
    pred_seg = seg_idx{i};
    
    % Get ground-truth label
    gt_lbl = tvsum50(i).user_anno;
    
    % Compute pairwise f1 scores
    ypred = solve_knapsack( pred_lbl, pred_seg, budget );
    ytrue = cell(size(gt_lbl, 2),1);
    for j = 1:size(gt_lbl,2)
        ytrue{j} = solve_knapsack(gt_lbl(:,j), pred_seg, budget );
        cp{j} = classperf( ytrue{j}, ypred, 'Positive', 1, 'Negative', 0 );
    end
    stats(i).ypred = ypred;
    stats(i).video  = tvsum50(i).video;
    
    stats(i).prec = cellfun(@(x) x.PositivePredictiveValue, cp);
    stats(i).rec  = cellfun(@(x) x.Sensitivity, cp);
    stats(i).f1 = max(0, 2*(stats(i).prec.*stats(i).rec)./(stats(i).prec + stats(i).rec));
    
    stats(i).mean_f1 = sum(stats(i).f1) / size(gt_lbl,2);
end
%{
%% Report results similar to Table 2 in CVPR 2015 paper
%
categories = {'VT','VU','GA','MS','PK','PR','FM','BK','BT','DS'};

stats_mean_f1 = cellfun(@(x) x.mean_f1, stats);
stats_method = cellfun(@(x) x.method, stats(:,1), 'UniformOutput',false);
stats_video  = cellfun(@(x) x.video, stats(1,:), 'UniformOutput',false);

per_category_perf = cell(1,numel(categories));
for i = 1:numel(categories),
    videos = {tvsum50(find(strcmp({tvsum50.category},categories{i}))).video};
    
    per_category_perf{i} = zeros(numel(stats_method),numel(videos));
    for j = 1:numel(videos),
        per_category_perf{i}(:,j) = stats_mean_f1(:,find(strcmp(stats_video,videos{j})));
    end
end

% Print out the table
fprintf('       ');
for i = 1:numel(stats_method),
    fprintf('[%s]  ', stats_method{i});
end
fprintf('\n');

for i = 1:numel(categories),
    fprintf('[%s]\t', categories{i});
    for j = 1:numel(stats_method),
        fprintf('%.2f  ', mean(per_category_perf{i}(j,:)));
    end
    fprintf('\n');
end

fprintf('[avg]\t');
for i = 1:numel(stats_method),
    fprintf('%.2f  ', mean(stats_mean_f1(i,:)));
end
fprintf('\n');
%}
