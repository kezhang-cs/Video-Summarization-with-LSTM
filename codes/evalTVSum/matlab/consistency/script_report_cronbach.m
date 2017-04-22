%{
Yahoo! TVSum50 Dataset.
- Script to compute label consistency
%}

load ../tvsum50.mat;
categories = unique( {tvsum50.category} );

consistency.cronbach_alpha = [];
consistency.pairwise_f1_avg = [];


%% Cronbach's Alpha
fprintf('Cronbach Alpha\n');
for i=1:numel(tvsum50),
    consistency.cronbach_alpha(i) = cronbach( tvsum50(i).user_anno );
end

for i=1:numel(categories),
    idx = find(strcmp({tvsum50.category},categories{i}));
    alpha = consistency.cronbach_alpha(idx);
    fprintf('[%s] %.2f (%.2f)\n', categories{i}, mean(alpha),std(alpha));
end
fprintf('[avg] %.2f (%.2f)\n', mean(consistency.cronbach_alpha),std(consistency.cronbach_alpha));


%% Pairwise F1
fprintf('Pairwise F1 scores\n');
consistency.pairwise_f1 = {};
for i=1:numel(tvsum50),
    consistency.pairwise_f1{i} = pairwise_f1( tvsum50(i).user_anno, 0.15 );
end
consistency.pairwise_f1_avg = cellfun(@(x) mean(x), consistency.pairwise_f1);

for i=1:numel(categories),
    idx = find(strcmp({tvsum50.category},categories{i}));
    f1 = consistency.pairwise_f1_avg(idx);
    fprintf('[%s] %.2f (%.2f)\n', categories{i}, mean(f1),std(f1));
end
fprintf('[avg] %.2f (%.2f)\n', mean(consistency.pairwise_f1_avg),std(consistency.pairwise_f1_avg));
