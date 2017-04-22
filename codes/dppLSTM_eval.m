function dppLSTM_eval( data_dir, dataset, file_name)
% evaluation dataset performance based on the packages.
% summ_type: 0 for importance-based and 1 for keyframe-based

file_dir = ['./res_LSTM/' file_name];
if exist(file_dir, 'file') ~= 2
    return
end

test_inds = [int32(h5read(file_dir,'/idx')) + 1]';
test_kernel = h5read(file_dir,'/pred');
test_pred = h5read(file_dir,'/pred_k');

videos_test = struct;
num_frm = load([data_dir 'cnt_frm_' dataset '.mat']);
num_frm_sample = load([data_dir 'cnt_frm_sample_' dataset '.mat']);
load([data_dir 'shot_' dataset '.mat']);
cnt_frm_sum = [0; cumsum(double(h5read(file_dir,'/cFrm')))];

for i = 1:length(test_inds)
    tid = test_inds(i);
    shot_boundary = shot_boundaries{tid};
    frm_sample  = num_frm_sample.segVid{tid};
    lstm_pred = test_pred(:, cnt_frm_sum(i)+1:cnt_frm_sum(i+1));
    lstm_qual = test_kernel(cnt_frm_sum(i)+1:cnt_frm_sum(i+1));
    lstm_pred = lstm_pred.*repmat(lstm_qual, size(lstm_pred,1),1);
    [videos_test(tid).Ypred] = gen_summ_key(lstm_pred, lstm_qual, frm_sample, shot_boundary, num_frm.cFrm(tid));
    videos_test(tid).Ypred = get_th(videos_test(tid).Ypred, shot_boundary);
end
[mean_res]= evaluation_user_idx(dataset, videos_test, test_inds);
fprintf('file name: %s f-score: %.3f\n', file_name, mean_res);

%% Start-up
function [mean_res] = evaluation_user_idx(dataset, videos, testInds)

if strcmp(dataset, 'TVSum')
    [res] = evaluation_user_TVSum(videos, testInds);
    mean_res = mean(res(:,1));
elseif strcmp(dataset, 'SumMe')
    [res] = evaluation_user_SumMe(videos, testInds);
    mean_res = mean(res(:,6));
end

function [res] = evaluation_user_TVSum(videos, test_inds)
cd('./evalTVSum/');
[stats] = evaluate_TVSum(videos, test_inds);
numTest = length(test_inds);
f_measure = zeros(1,numTest); recall = zeros(1,numTest); prec = zeros(1,numTest); summary_length = zeros(1,numTest); res = zeros(1,numTest);
for i = 1: length(test_inds)
    tid = test_inds(i);
    f_measure(i) = stats(tid).mean_f1; recall(i) = mean(stats(tid).rec); prec(i) = mean(stats(tid).prec); summary_length(i) = sum(stats(tid).ypred)/length(stats(tid).ypred);
end
res = [f_measure' recall' prec' summary_length'];
cd ..

function [res] = evaluation_user_SumMe(videos_test, te_ind)
cd('./evalSumMe/')
load('SumMe_data.mat');
HOMEDATA='./GT/';
f_measure = zeros(length(te_ind),1);
summary_length = zeros(length(te_ind),1);
recall = zeros(length(te_ind),1);
prec = zeros(length(te_ind),1);
f_measure_seg = zeros(length(te_ind),1);

f_measure_m = zeros(length(te_ind),1);
summary_length_m = zeros(length(te_ind),1);
recall_m = zeros(length(te_ind),1);
prec_m = zeros(length(te_ind),1);
f_measure_seg_m = zeros(length(te_ind),1);

for i = 1: length(te_ind)
    tid = te_ind(i);
    [f_measure(i), summary_length(i), recall(i), prec(i), ~, f_measure_seg(i)] = ...
        summe_evaluateSummary(videos_test(tid).Ypred, nameVid{tid}, HOMEDATA, -1, 1);
    [f_measure_m(i), summary_length_m(i), recall_m(i), prec_m(i), ~, f_measure_seg_m(i)] = ...
        summe_evaluateSummary_m(videos_test(tid).Ypred, nameVid{tid}, HOMEDATA, -1, 1);
end
res = [f_measure recall prec summary_length f_measure_seg f_measure_m recall_m prec_m summary_length_m f_measure_seg_m];
cd ..

%% auxiliary functions

function Ypred = get_th(frame_score, shot_boundaries)
eval_th = 0.15;
seg_tmp = zeros(numel(shot_boundaries)-1,2);
for j=1:numel(shot_boundaries)-1,
    seg_tmp(j,2) = shot_boundaries(j+1);
    seg_tmp(j,1) = shot_boundaries(j)+1;
end
seg_idx = seg_tmp;
Ypred = get_shot_knapsack(frame_score, seg_idx, eval_th );

function Ypred = proj_sub_shot(Ypred_frm, seg_vid_frm, seg_vid_shot, cnt_frm)

newArr = zeros(cnt_frm,1); 
for j = 1:length(seg_vid_frm)-1
    newArr(seg_vid_frm(j)+1:seg_vid_frm(j+1)) = Ypred_frm(j);
end
Ypred = zeros(1, length(seg_vid_shot)-1); 
for j = 1:length(seg_vid_shot)-1
    Ypred(j) = mean(newArr(seg_vid_shot(j)+1:seg_vid_shot(j+1)));
end

function [Ypred_test] = gen_summ_key(lstm_pred, lstm_qual, seg_vid_frm, seg_vid_shot, cnt_frm) 
L = lstm_pred'*lstm_pred;
Ypred = diag(L);
Ypred(predictY_inside(L)) = Ypred(predictY_inside(L)) + max(lstm_qual); 
Ypred = proj_sub_shot(Ypred, seg_vid_frm, seg_vid_shot, cnt_frm);
Ypred_test = zeros(seg_vid_shot(end),1);
for j = 1: length(seg_vid_shot)-1
    Ypred_test([seg_vid_shot(j)+1: seg_vid_shot(j+1)]) = Ypred(j);
end

function [Y_record, fVal, L] = predictY_inside(L)
L = normalizeKernel(L); 
Y_record = [];
fVal = -1e10;

% new greedy goes here
Y_record = greedy_sym(L);

function S = greedy_sym(L)
% Initialize.
N = size(L,1);
S = [];

% Iterate through items.
for i = 1:N
    lift0 = obj(L,[S i]) - obj(L,S);
    lift1 = obj(L,[S (i+1):N]) - obj(L,[S i:N]);
    
    if lift0 > lift1
        S = [S i];
    end
end

function f = obj(L, S)
f = log(det(L(S, S)));

function L = normalizeKernel(nL)
L = (nL+nL')/2';

[eigVec,eigVal] = eig(L);
eigVec = real(eigVec);
eigVec = eigVec./(ones(size(eigVec, 1), 1)*sqrt(sum(eigVec.^2, 1)));
eigVal = real(eigVal);
eigVal(eigVal<0) = 0;
L = eigVec*eigVal*eigVec';