function [user_iou_recall,user_iou_precision] = summe_evaluateIOU(summary_selection,user_score)
%%
  OverlapTh=0.25;

  % Check length
  nFrames=length(user_score);
  if size(summary_selection,1)==1
       summary_selection=summary_selection';
  end
  if length(summary_selection) < nFrames
       warning('Pad selection with %d zeros!',nFrames-length(summary_selection) )
       summary_selection(end+1:nFrames)=0;
  elseif length(summary_selection) > nFrames
       warning('Crop selection (%d frames) to GT length',length(summary_selection)-nFrames )
       summary_selection=summary_selection(1:nFrames);
  end
          
  
  %% Compute IOU of segments    
  % 1) Build segments    
  isInSel=0;
  currSeg=0;
  auto_segments=zeros(numel(summary_selection),1);
  for frameIdx=1:numel(summary_selection)
      if summary_selection(frameIdx)>0
          if ~isInSel
              currSeg=currSeg+1;
              isInSel=true;
          end
          auto_segments(frameIdx)=currSeg;
          
      else
          isInSel=false;
      end       
  end
  
  % 2) precision
  user_iou=0;
  segIOU=[];
  for autoSegIdx=unique(auto_segments)'
    tmp_iou=0;      
    if autoSegIdx==0
         continue;
     end           
    for userSegIdx=unique(user_score)'          
         if userSegIdx==0
              continue;
         end              
        intersection=sum((user_score==userSegIdx).*(auto_segments==autoSegIdx));
        if intersection==0
            continue;
        end
        union=sum((user_score==userSegIdx)+(auto_segments==autoSegIdx));       
        if union>0
            tmp_iou(end+1)=intersection/union;
        else
            error('This cannot happen')
        end
    end
    segIOU(end+1)=max(tmp_iou);
  end
  if exist('OverlapTh','var')
      segIOU=segIOU>=OverlapTh;
  end
  user_iou_precision=mean(segIOU);
  
  % 3) recall
  user_iou=0;
  segIOU=[];
  for userSegIdx=unique(user_score)'       
      tmp_iou=0;  
     if userSegIdx==0
          continue;
     end         
    for autoSegIdx=unique(auto_segments)'    
        if autoSegIdx==0
             continue;
        end           
                  
        intersection=sum((user_score==userSegIdx).*(auto_segments==autoSegIdx));
        if intersection==0
            continue;
        end
        union=sum((user_score==userSegIdx)+(auto_segments==autoSegIdx));       
        if union>0
            tmp_iou(end+1)=intersection/union;
        else
            error('This cannot happen')
        end
    end
    segIOU(end+1)=max(tmp_iou);
  end
  if exist('OverlapTh','var')
      segIOU=segIOU>=OverlapTh;
  end
  user_iou_recall=mean(segIOU);  
end
