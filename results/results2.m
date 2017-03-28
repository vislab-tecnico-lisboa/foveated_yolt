close all
addpath('export_fig');

gt_folder='../dataset/gt/';

detections_file='../dataset/detections/raw_bbox_parse_high_blur_caffenet_100.txt';   % raw_bbox_parse_caffenet_100.txt
feedback_detections_file = '../dataset/detections/feedback_detection_parse_high_blur_caffenet_100.txt';

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
[sigmas,threshs,classes,scores]=parse_detections2(...
    images_number,...
    detections_file);

% get feedback detections 
[feedback_sigmas,feedback_threshs,feedback_classes,feedback_scores,rank_feedback_classes]=feedback_parse_detections2(...
    images_number,...
    feedback_detections_file);




% %% LOCALIZATION
% % get detection error rates (YOLO)
% [detection_error_rate] = detection_error_rates(sigmas,threshs,images_number,detections,gt_detections,detections_resolution,top_k,overlap_correct);
% 
% % get detection crop error rate
% [detection_crop_error_rate] = detection_error_rates(feedback_crop_sigmas,feedback_crop_threshs,images_number,feedback_crop_detections,gt_detections,detections_resolution,top_k,overlap_correct);
% 
% % get detection foveate error rate
% [detection_foveate_error_rate] = detection_error_rates(feedback_sigmas,feedback_threshs,images_number,feedback_detections,gt_detections,detections_resolution,top_k,overlap_correct);


%% CLASSIFICATION
% get classification error rates (YOLO)
[top1_classification_error_rate, top5_classification_error_rate] = classification_error_rates(sigmas,threshs,images_number,classes,gt_classes,top_k);

% get feedback classification error rates (YOLT foveation)
[top1_feedback_classification_error_rate, top5_feedback_classification_error_rate] = classification_error_rates(feedback_sigmas,feedback_threshs,images_number,rank_feedback_classes,gt_classes,top_k);





%% classification (top 1) error plots

% fix one threshold and plot all sigmas
thresh_index=1;
thresh_crop_index=15;

classification_legend = {...
    char('top 1 ');...
    char('top 5 ');...
    char('top 1 feedback ');...
    char('top 5 feedback ');... 
    };


figure(1)
fontsize=15;
set(gcf, 'Color', [1,1,1]);
% plot(sigmas,100*repmat(top1_classification_error_rate(:,thresh_index),length(feedback_sigmas)),'g-*');
% hold on
% plot(sigmas,100*repmat(top5_classification_error_rate(:,thresh_index),length(feedback_sigmas)),'g-o');

plot(sigmas,100*top1_classification_error_rate(:,:),'r-*'); 
hold on
plot(sigmas,100*top5_classification_error_rate(:,:),'r-o'); 

plot(feedback_sigmas,100*top1_feedback_classification_error_rate(:,:),'b--*'); 
plot(feedback_sigmas,100*top5_feedback_classification_error_rate(:,:),'b--o'); 


xlabel('$\sigma$','Interpreter','LaTex','FontSize',fontsize);
ylabel('Classification Error (%)','Interpreter','LaTex','FontSize',fontsize);
ylim([0 100])
%legend('show', 'DislpayName', classification_legend(:) ,'Location', 'best');
legend(classification_legend(:),'Location', 'southeast');
set(gcf, 'PaperPosition', [0 0 200 100]); %Position the plot further to the left and down. Extend the plot to fill entire paper.
set(gcf, 'PaperSize', [200 100]); %Keep the same paper size
%saveas(gcf, 'test', 'pdf')
saveas(figure(1),'classification_error_high_blur_caffenet_100.pdf')
%export_fig localization_error_sigma -pdf



