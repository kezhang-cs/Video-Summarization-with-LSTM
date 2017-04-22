function [f_measure, summary_length,recall, p, nn_f_meas, f_measure_seg, maxid] = summe_evaluateSummary_m(summary_selection,videoName,HOMEDATA,testIdx,evalIOU)
%%[f_measure, summary_length,r] = summe_evaluateSummary(summary_selection,videoName,HOMEDATA,testIdx,evalIOU)
% Evaluates a summary for video videoName 
% (where HOMEDATA points to the ground truth directory)
% f_measure is the mean pairwise f-measure used in Gygli et al. ECCV 2014

  %% Load GT file
  load(fullfile(HOMEDATA,[videoName '.mat']),'user_score','nFrames');
  nFrames=size(user_score,1);
  nbOfUsers=size(user_score,2);
  
  %% Check inputs
  if ~exist('testIdx','var')
      testIdx=-1;
  end
  if size(summary_selection,1)==1
       summary_selection=summary_selection';
  end
  if ~exist('evalIOU','var')
      evalIOU=false;
  end
  if length(summary_selection) < nFrames
       warning('Pad selection with %d zeros!',nFrames-length(summary_selection) )
       summary_selection(end+1:nFrames)=0;
  elseif length(summary_selection) > nFrames
       warning('Crop selection (%d frames) to GT length',length(summary_selection)-nFrames )
       summary_selection=summary_selection(1:nFrames);
  end
          
  
  %% Compute pairwise f-measure, summary length and recall          
  user_intersection=zeros(1,nbOfUsers);
  for userIdx=1:nbOfUsers;
      user_intersection(userIdx)=nnz(user_score(:,userIdx).*summary_selection);
      
      if evalIOU
        [recall_seg(userIdx),precision_seg(userIdx)]=summe_evaluateIOU(summary_selection,user_score(:,userIdx));
      end      
  end
 
  recall=user_intersection([1:nbOfUsers]~=testIdx)./sum(user_score(:,[1:nbOfUsers]~=testIdx)>0);
  p=user_intersection([1:nbOfUsers]~=testIdx)./nnz(summary_selection);
  f_measure=2*recall.*p./(recall+p);
  f_measure(isnan(f_measure))=0;
  nn_f_meas=max(f_measure);
  [f_measure, maxid] = max(f_measure); % modification here 
  
  if evalIOU
      f_measure_seg=2*recall_seg.*precision_seg ./ (recall_seg+precision_seg);
      f_measure_seg(isnan(f_measure_seg))=0;
      f_measure_seg=max(f_measure_seg); % modification here 
  else
      f_measure_seg=-1;
  end
    
  if isnan(f_measure)
      f_measure=0;
  end
  summary_length=nnz(summary_selection>0)./length(summary_selection);
    
  recall=recall(maxid);% modification here 
  p = p(maxid);% modification here 
end
