function [error_rate] = detection_error_rates(sigmas,threshs,images_number,detections,gt_detections,detections_resolution,top_k,overlap_correct)

% check overlaps
overlaps=[];
for s=1:length(sigmas)
    for t=1:length(threshs)
        for i=1:images_number
            gt_size=gt_detections(i).size;
            aspect_ratio_x=gt_size(1)/detections_resolution;
            aspect_ratio_y=gt_size(2)/detections_resolution;
            % for each detection (of the 5)
            for j=1:top_k
                % scale detections
                detection=reshape(detections(s,t,i,j,:),1,4);
                detection(1)=detection(1)*aspect_ratio_x;
                detection(2)=detection(2)*aspect_ratio_y;
                detection(3)=detection(3)*aspect_ratio_x;
                detection(4)=detection(4)*aspect_ratio_y;
                % check overlaps for each ground truth bounding box
                overlaps(s,t,i,j).overlap=zeros(size(gt_detections(i).bboxes,1),1);
                
                for g=1:size(gt_detections(i).bboxes,1)
                    % gt bbox
                    gt_bbox=gt_detections(i).bboxes(g,:);
                    
                    overlaps(s,t,i,j).overlap(g)=bboxOverlapRatio(gt_bbox,detection);
                    
                end
            end
        end
    end
end

% consider only the maximum overlap over ground truths
max_overlap=zeros(length(sigmas),length(threshs), images_number,top_k);
for s=1:length(sigmas)
    for t=1:length(threshs)
        for i=1:images_number
            for j=1:top_k
                max_overlap(s,t,i,j)=max(overlaps(s,t,i,j).overlap);
            end
        end
    end
end

% consider only the maximum overlap over all 5 detections
max_overlap=max(max_overlap,[],4);

% evaluate if the detections were good
max_overlap(max_overlap>=overlap_correct)=1;
max_overlap(max_overlap<overlap_correct)=0;

% compute detection rate
error_rate=(size(max_overlap,3)-sum(max_overlap,3))/size(max_overlap,3);