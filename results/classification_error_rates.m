function [top1_classification_error_rate, top5_classification_error_rate] = classification_error_rates(sigmas,threshs,images_number,classes,gt_classes, top_k)

top1_classification=zeros(length(sigmas),length(threshs),images_number);
top5_classification=zeros(length(sigmas),length(threshs),images_number);

for s=1:length(sigmas)
    for t=1:length(threshs)
        for i=1:images_number
            gt_class=gt_classes{i};
            top1_classification(s,t,i)=strcmpi(strtrim(classes{s,t,i,1}),gt_class);  
            % for each detection (of the 5)
            for j=1:top_k
                top5_classification(s,t,i)= (strcmpi(strtrim(classes{s,t,i,j}),gt_class) | top5_classification(s,t,i));
            end
        end
    end
end

% compute classification error rate
top1_classification_error_rate=(size(top1_classification,3)-sum(top1_classification,3))/size(top1_classification,3);
top5_classification_error_rate=(size(top5_classification,3)-sum(top5_classification,3))/size(top5_classification,3);