%{
Yahoo! TVSum50 Dataset.
- Function to solve the 0/1 Knapsack problem
%}

function [ out ] = solve_knapsack( frame_scores, segments, portion )
%SOLVE_KNAPSACK Summary of this function goes here

    n_shots = size(segments,1);
    shot_scores = zeros(n_shots,1);
    for i = 1:n_shots,
        range = segments(i,1):min(numel(frame_scores),segments(i,2));
        shot_scores(i) = mean(frame_scores(range));
    end
    
    shot_weights = (segments(:,2)-segments(:,1)+1)';
    budget = fix(portion * numel(frame_scores));

    [~,amount] = knapsack(shot_weights,shot_scores,budget);
    
    out_shot = amount;
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
end

