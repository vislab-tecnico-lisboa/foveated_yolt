function [tp1_class_error_av,tp1_class_error_std,tp5_class_error_av,tp5_class_error_std] = classification_error_rates(sigmas,threshs,fix_pts, images_number,classes,gt_classes, top_k)

top1_classification=zeros(length(sigmas),length(threshs),size(fix_pts,1),images_number);
top5_classification=zeros(length(sigmas),length(threshs),size(fix_pts,1),images_number);

for s=1:length(sigmas)
    for t=1:length(threshs)
        for p=1:size(fix_pts,1)
            for im=1:images_number
                gt_class=gt_classes{im};
                top1_classification(s,t,p,im)=strcmpi(strtrim(classes{s,t,p,im,1}),gt_class); 
                % for each detection (of the 5)
                for j=1:top_k
                    top5_classification(s,t,p,im)=(strcmpi(strtrim(classes{s,t,p,im,j}),gt_class) | top5_classification(s,t,p,im));
                end
            end
        end
    end
end

top1_classification_error_rate=((size(top1_classification,4))-...
    sum(top1_classification,4))/(size(top1_classification,4));
tp1_class_error_av = mean(top1_classification_error_rate,3);
tp1_class_error_std = std(top1_classification_error_rate,1,3);

top5_classification_error_rate=((size(top5_classification,4))-...
    sum(top5_classification,4))/(size(top5_classification,4));
tp5_class_error_av = mean(top5_classification_error_rate,3);
tp5_class_error_std = std(top5_classification_error_rate,1,3);
