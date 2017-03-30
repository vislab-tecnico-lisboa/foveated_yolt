close all
addpath('export_fig');

gt_folder='../dataset/gt/';

detections_file='../dataset/detections/new/raw_bbox_parse_crop_high_blur_caffenet_100.txt';   % raw_bbox_parse_caffenet_100.txt
feedback_detections_file = '../dataset/detections/new/feedback_detection_parse_fovea_high_blur_caffenet_100.txt';
feedback_crop_detections_file = '../dataset/detections/new/feedback_detection_parse_crop_high_blur_caffenet_100.txt';

classifications_file='../files/ground_truth_labels_ilsvrc12.txt';
images_folder='../dataset/images/';

% set to true to check detections
view_detections=false;

% parameters
detections_resolution=227;
images_number=100; 
overlap_correct=0.5;
top_k=5;


%% DETECTIONS

% get ground truth
[gt_detections, gt_classes]=parse_ground_truth(gt_folder,classifications_file,images_number);

% get detections (YOLO)
[sigmas,threshs,classes,scores,detections]=parse_detections2(...
    images_number,...
    detections_file);

% get feedback detections (Fovea)
[feedback_sigmas,feedback_threshs,feedback_classes,feedback_scores,rank_feedback_classes,feedback_detections]=feedback_parse_detections2(...
    images_number,...
    feedback_detections_file);

% get feedback detections (Crop)
[feedback_crop_sigmas,feedback_crop_threshs,feedback_crop_classes,feedback_crop_scores,rank_feedback_crop_classes,feedback_crop_detections]=feedback_parse_detections2(...
    images_number,...
    feedback_crop_detections_file);



%% LOCALIZATION
% get detection error rates (YOLO)
[detection_error_rate] = detection_error_rates(sigmas,threshs,images_number,detections,gt_detections,detections_resolution,top_k,overlap_correct);

% % get detection crop error rate (CROP)
[detection_crop_error_rate] = detection_error_rates(feedback_crop_sigmas,feedback_crop_threshs,images_number,feedback_crop_detections,gt_detections,detections_resolution,top_k,overlap_correct);
% 
% % get detection foveate error rate  (FOVEA)
[detection_foveate_error_rate] = detection_error_rates(feedback_sigmas,feedback_threshs,images_number,feedback_detections,gt_detections,detections_resolution,top_k,overlap_correct);


%% CLASSIFICATION
% get classification error rates (YOLO)
[top1_classification_error_rate, top5_classification_error_rate] = classification_error_rates(sigmas,threshs,images_number,classes,gt_classes,top_k);

% get feedback classification error rates (Fovea)
[top1_feedback_classification_error_rate, top5_feedback_classification_error_rate] = classification_error_rates(feedback_sigmas,feedback_threshs,images_number,rank_feedback_classes,gt_classes,top_k);

% get feedback classification error rates (crop)
[top1_feedback_crop_classification_error_rate, top5_feedback_crop_classification_error_rate] = classification_error_rates(feedback_crop_sigmas,feedback_crop_threshs,images_number,rank_feedback_crop_classes,gt_classes,top_k);





%% CLASSIFICATION ERROR PLOTS

classification_legend = {...
    char('top 1 ');...
    char('top 5 ');...
    char('top 1 feedback (fovea) ');...
    char('top 5 feedback (fovea) ');... 
    char('top 1 feedback (crop) ');...
    char('top 5 feedback (crop) ');...
    };


figure(1)
fontsize=15;
set(gcf, 'Color', [1,1,1]);

plot(sigmas,100*top1_classification_error_rate(:,2),'k-*'); 
hold on
plot(sigmas,100*top5_classification_error_rate(:,2),'k-o'); 

plot(feedback_sigmas,100*top1_feedback_classification_error_rate(:,3),'b--*'); 
plot(feedback_sigmas,100*top5_feedback_classification_error_rate(:,3),'b--o'); 

plot(feedback_crop_sigmas,100*top1_feedback_crop_classification_error_rate(:,4),'r--*'); 
plot(feedback_crop_sigmas,100*top5_feedback_crop_classification_error_rate(:,4),'r--o'); 


xlabel('$\sigma$','Interpreter','LaTex','FontSize',fontsize);
ylabel('Classification Error (%)','Interpreter','LaTex','FontSize',fontsize);
ylim([0 100])
%legend('show', 'DislpayName', classification_legend(:) ,'Location', 'best');
legend(classification_legend(:),'Location', 'southeast');
set(gcf, 'PaperPosition', [0 0 200 100]);   %Position the plot further to the left and down. Extend the plot to fill entire paper.
set(gcf, 'PaperSize', [200 100]);           %Keep the same paper size
saveas(figure(1),'classification_error_high_blur_caffenet_100.pdf')
%export_fig localization_error_sigma -pdf



%% LOCALIZATION ERROR PLOTS

localizaion_legend = {...
    char('YOLO');...    
    char('YOLT (fovea)');...
    char('YOLT (crop)');...
    };


figure(2)
fontsize=15;
set(gcf, 'Color', [1,1,1]);

plot(sigmas,100*detection_error_rate(:,3),'k-o');           % first pass
hold on
plot(sigmas,100*detection_foveate_error_rate(:,3),'b-o');   % fovea

plot(sigmas,100*detection_crop_error_rate(:,3),'r-o');      % crop


xlabel('$\sigma$','Interpreter','LaTex','FontSize',fontsize);
ylabel('Localization Error (%)','Interpreter','LaTex','FontSize',fontsize);
ylim([0 100])
%legend('show', 'DislpayName', classification_legend(:) ,'Location', 'best');
legend(localizaion_legend(:),'Location', 'southeast');
set(gcf, 'PaperPosition', [0 0 200 100]);   %Position the plot further to the left and down. Extend the plot to fill entire paper.
set(gcf, 'PaperSize', [200 100]);           %Keep the same paper size
saveas(figure(2),'localization_error_high_blur_caffenet_100.pdf')
%export_fig localization_error_sigma -pdf


