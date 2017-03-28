%{
Yahoo! TVSum50 Dataset.
- Function to compute a pairwise F1 score
%}

function [ out ] = pairwise_f1( score,portion )
%PAIRWISE_F1 Summary of this function goes here

    addpath('../knapsack');
    
    if size(score,1) < size(score,2),
        error('score must be column-wise matrix; each column is an observation');
    end
    
    L = size(score,1); % sequence length
    N = size(score,2); % number of annotations
    
    % Pre-compute knapsack solution
    y = cell(1,N);
    for i=1:N
        [~,y{i}] = knapsack(ones(L,1), score(:,i), fix(portion*L));
    end
    
    for i=1:N,
        val = 0; % accumulative f-measure
        for j=1:N,
            if i==j, continue; end
            cp = classperf(y{i},y{j},'Positive',1,'Negative',0);
            prec = cp.CorrectRate;
            rec = cp.Sensitivity;
            f1score = 2*(prec*rec)/(prec+rec);

            val = val+f1score;
        end
        val = val / (N-1);
        out(i) = val;
    end
end
