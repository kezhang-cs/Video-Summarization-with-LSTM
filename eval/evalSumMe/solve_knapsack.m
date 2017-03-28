%{
Yahoo! TVSum50 Dataset.
- Function to solve the 0/1 Knapsack problem
%}

function [ out_shot ] = solve_knapsack( shot_scores, segVid, portion, cFrm)
%SOLVE_KNAPSACK Summary of this function goes here
addpath('./knapsack');

shotBoundaries = segVid;

segments = zeros(numel(shotBoundaries)-1,2);
for j=1:numel(shotBoundaries)-1,
    segments(j,2) = shotBoundaries(j+1);
    segments(j,1) = shotBoundaries(j)+1;
end

shot_weights = (segments(:,2)-segments(:,1)+1)';
budget = fix(portion * cFrm);

[~,amount] = knapsack(shot_weights,shot_scores,budget);

out_shot = amount;
%{
out = zeros(size(frame_scores));
for i=1:numel(out_shot),
    if out_shot(i)==1,
        range = segments(i,1):min(numel(frame_scores),segments(i,2));
        out(range) = 1;
    end
end

if unique(out)==0,
    out = frame_scores > quantile(frame_scores, 1-portion);
end

if unique(out)==0,
    i=1;
    [~,ind] = sort(frame_scores,'descend');
    while sum(out)/numel(out) < portion,
        out(ind(i))=1;
        i = i+1;
    end
end
%}
end

