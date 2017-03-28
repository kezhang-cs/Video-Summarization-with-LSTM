%{
Yahoo! TVSum50 Dataset.
- Script to evaluate summarization result
%}

clear all; clc;

addpath('./knapsack');
load ../tvsum50.mat; % load the dataset

n_videos = numel(tvsum50);
budget = 0.15;

%% Randomly generate frame-level scores
%
results = cell(4,n_videos);
for i=1:n_videos,
    
    Fc = 1:tvsum50(i).nframes;
    
    results{1,i}.method = 'rnd';
    results{1,i}.frame_score = rand(tvsum50(i).nframes,1);
    
    results{2,i}.method = 'sin';
    results{2,i}.frame_score = sin(Fc./300)';
    
    results{3,i}.method = 'tan';
    results{3,i}.frame_score = tan(Fc./300)';
    
    results{4,i}.method = 'inv';
    results{4,i}.frame_score = -(1:tvsum50(i).nframes)';
end

% [0 1] normalize
for i=1:size(results,1),
    for j=1:size(results,2),
        results{i,j}.frame_score = results{i,j}.frame_score - min(results{i,j}.frame_score);
        results{i,j}.frame_score = results{i,j}.frame_score / max(results{i,j}.frame_score);
    end
end


%% Generate segmentation results
%
shot_length = 50; % each shot has 50 frames
segment_results = cell(1,n_videos);
for i=1:n_videos,
    shot_boundaries = shot_length:shot_length:tvsum50(i).nframes;
    
    S = ones(numel(shot_boundaries),2);
    for j=1:numel(shot_boundaries)-1,
        S(j,2) = shot_boundaries(j);
        S(j+1,1) = shot_boundaries(j)+1;
    end
    S(numel(shot_boundaries),2) = tvsum50(i).nframes;
    
    segment_results{i} = S;
end


%% Evaluate performance
%
stats = cell(size(results));
for i = 1:size(results,1) 
    
    fprintf('Computing prediction performance of [ %s ]\n', results{i,1}.method);
    
    % 50 videos
    for j = 1:size(results,2) 
        % Get prediction label
        pred_lbl = results{i,j}.frame_score;
        
        % Get predicted shot segments
        pred_seg = segment_results{j};
        
        % Get ground-truth label
        gt_lbl = tvsum50(j).user_anno;
        
        % Compute pairwise f1 scores
        ypred = solve_knapsack( pred_lbl, pred_seg, budget );
        for k = 1:size(gt_lbl,2),
            ytrue{k} = solve_knapsack( gt_lbl(:,k), pred_seg, budget );
            cp{k} = classperf( ytrue{k}, ypred, 'Positive', 1, 'Negative', 0 );
        end
        
        stats{i,j}.video  = tvsum50(j).video;
        stats{i,j}.method = results{i,j}.method;
        
        stats{i,j}.prec = cellfun(@(x) x.CorrectRate, cp);
        stats{i,j}.rec  = cellfun(@(x) x.Sensitivity, cp);
        stats{i,j}.f1 = 2*(stats{i,j}.prec.*stats{i,j}.rec)./(stats{i,j}.prec+stats{i,j}.rec);
        
        stats{i,j}.mean_f1 = sum(stats{i,j}.f1) / size(gt_lbl,2);
    end
end


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

