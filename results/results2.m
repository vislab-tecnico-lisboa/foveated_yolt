close all
addpath('export_fig');

gt_folder='../dataset/gt/';

detections_file_base_localization='../dataset/detections/new/raw_bbox_parse_local_yolo_high_resolution_caffenet_1000.txt';

% Caffe
detections_file='../dataset/detections/new/raw_bbox_parse_crop_high_blur_caffenet_100.txt';
feedback_detections_file = '../dataset/detections/new/feedback_detection_parse_fovea_high_blur_caffenet_100.txt';
feedback_crop_detections_file = '../dataset/detections/new/feedback_detection_parse_crop_high_blur_caffenet_100.txt';

% VGG
detections_vgg_file = '../dataset/detections/new/raw_bbox_parse_crop_high_blur_vggnet_100.txt';
feedback_detections_vgg_file = '../dataset/detections/new/feedback_detection_parse_fovea_high_blur_vggnet_100.txt';
feedback_crop_vgg_detections_file = '../dataset/detections/new/feedback_detection_parse_crop_high_blur_vggnet_100.txt';

% Google
detections_google_file = '../dataset/detections/new/raw_bbox_parse_fovea_high_blur_google_100.txt';
feedback_detections_google_file = '../dataset/detections/new/feedback_detection_parse_fovea_high_blur_google_100.txt';
feedback_crop_google_detections_file = '../dataset/detections/new/feedback_detection_parse_crop_high_blur_google_100.txt';


% Ground truth
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

% get detections (YOLO) - Caffe
[sigmas,threshs,classes,scores,detections]=parse_detections2(...
    images_number,...
    detections_file);

% get detections (YOLO) - VGG
[vgg_sigmas,vgg_threshs,vgg_classes,vgg_scores,vgg_detections]=parse_detections2(...
    images_number,...
    detections_vgg_file);

% get detections (YOLO) - Google
[google_sigmas,google_threshs,google_classes,google_scores,google_detections]=parse_detections2(...
    images_number,...
    detections_google_file);

% get detections (best localization base line)
[local_sigmas,local_threshs,local_classes,local_scores,local_detections]=parse_detections2(...
    images_number,...
    detections_file_base_localization);

% CAFFE
% get feedback detections (Fovea) 
[feedback_sigmas,feedback_threshs,feedback_classes,feedback_scores,rank_feedback_classes,feedback_detections]=feedback_parse_detections2(...
    images_number,...
    feedback_detections_file);

% get feedback detections (Crop) 
[feedback_crop_sigmas,feedback_crop_threshs,feedback_crop_classes,feedback_crop_scores,rank_feedback_crop_classes,feedback_crop_detections]=feedback_parse_detections2(...
    images_number,...
    feedback_crop_detections_file);

% VGG
% get feedback detections (Fovea) 
[vgg_feedback_sigmas,vgg_feedback_threshs,vgg_feedback_classes,vgg_feedback_scores,vgg_rank_feedback_classes,vgg_feedback_detections]=feedback_parse_detections2(...
    50,...
    feedback_detections_vgg_file);

% get feedback detections (Crop) 
[vgg_feedback_crop_sigmas,vgg_feedback_crop_threshs,vgg_feedback_crop_classes,vgg_feedback_crop_scores,vgg_rank_feedback_crop_classes,vgg_feedback_crop_detections]=feedback_parse_detections2(...
    images_number,...
    feedback_crop_vgg_detections_file);


% Google
% get feedback detections (Fovea) 
[google_feedback_sigmas,google_feedback_threshs,google_feedback_classes,google_feedback_scores,google_rank_feedback_classes,google_feedback_detections]=feedback_parse_detections2(...
    images_number,...
    feedback_detections_google_file);

% get feedback detections (Crop) 
[google_feedback_crop_sigmas,google_feedback_crop_threshs,google_feedback_crop_classes,google_feedback_crop_scores,google_rank_feedback_crop_classes,google_feedback_crop_detections]=feedback_parse_detections2(...
    images_number,...
    feedback_crop_google_detections_file);


%% LOCALIZATION
% get detection error rates (YOLO)
%[detection_error_rate] = detection_error_rates(sigmas,threshs,images_number,detections,gt_detections,detections_resolution,top_k,overlap_correct);

% get detection error rates (localization base line) -  first pass, high
% resolution image
%[detection_local_error_rate] =  detection_error_rates(local_sigmas,local_threshs,images_number,local_detections,gt_detections,detections_resolution,top_k,overlap_correct);

% % get detection crop error rate (CROP)
%[detection_crop_error_rate] = detection_error_rates(feedback_crop_sigmas,feedback_crop_threshs,images_number,feedback_crop_detections,gt_detections,detections_resolution,top_k,overlap_correct);
 
% % get detection foveate error rate  (FOVEA)
%[detection_foveate_error_rate] = detection_error_rates(feedback_sigmas,feedback_threshs,images_number,feedback_detections,gt_detections,detections_resolution,top_k,overlap_correct);


%% CLASSIFICATION
% get classification error rates (YOLO) - Caffe
[top1_classification_error_rate, top5_classification_error_rate] = classification_error_rates(sigmas,threshs,images_number,classes,gt_classes,top_k);

% get feedback classification error rates (Fovea)
[top1_feedback_classification_error_rate, top5_feedback_classification_error_rate] = classification_error_rates(feedback_sigmas,feedback_threshs,images_number,rank_feedback_classes,gt_classes,top_k);

% get feedback classification error rates (crop)
[top1_feedback_crop_classification_error_rate, top5_feedback_crop_classification_error_rate] = classification_error_rates(feedback_crop_sigmas,feedback_crop_threshs,images_number,rank_feedback_crop_classes,gt_classes,top_k);



% get classification error rates (YOLO) - VGG
[top1_vgg_classification_error_rate, top5_vgg_classification_error_rate] = classification_error_rates(vgg_sigmas,vgg_threshs,images_number,vgg_classes,gt_classes,top_k);

% get feedback classification error rates (Fovea)
[top1_vgg_feedback_classification_error_rate, top5_vgg_feedback_classification_error_rate] = classification_error_rates(vgg_feedback_sigmas,vgg_feedback_threshs,50,vgg_rank_feedback_classes,gt_classes,top_k);

% get feedback classification error rates (crop)
[top1_vgg_feedback_crop_classification_error_rate, top5_vgg_feedback_crop_classification_error_rate] = classification_error_rates(vgg_feedback_crop_sigmas,vgg_feedback_crop_threshs,images_number,vgg_rank_feedback_crop_classes,gt_classes,top_k);


% get classification error rates (YOLO) - Google
[top1_google_classification_error_rate, top5_google_classification_error_rate] = classification_error_rates(google_sigmas,google_threshs,images_number,google_classes,gt_classes,top_k);

% get feedback classification error rates (Fovea)
[top1_google_feedback_classification_error_rate, top5_google_feedback_classification_error_rate] = classification_error_rates(google_feedback_sigmas,google_feedback_threshs,images_number,google_rank_feedback_classes,gt_classes,top_k);

% get feedback classification error rates (crop)
[top1_google_feedback_crop_classification_error_rate, top5_google_feedback_crop_classification_error_rate] = classification_error_rates(google_feedback_crop_sigmas,google_feedback_crop_threshs,images_number,google_rank_feedback_crop_classes,gt_classes,top_k);




%% CLASSIFICATION ERROR PLOTS - CaffeNet

classification_legend = {...
    char('top 1 ');...
    char('top 5 ');...
    char('top 1 feedback (fovea) ');...
    char('top 5 feedback (fovea) ');... 
    char('top 1 feedback (crop) ');...
    char('top 5 feedback (crop) ');...
    };


figure(1)
fontsize=20;
set(gcf, 'Color', [1,1,1]);

plot(sigmas,100*top1_classification_error_rate(:,3),'k--o'); 
hold on
plot(sigmas,100*top5_classification_error_rate(:,3),'k-o'); 

plot(feedback_sigmas,100*top1_feedback_classification_error_rate(:,3),'b--s'); 
plot(feedback_sigmas,100*top5_feedback_classification_error_rate(:,3),'b-s'); 

plot(feedback_crop_sigmas,100*top1_feedback_crop_classification_error_rate(:,4),'m--^'); 
plot(feedback_crop_sigmas,100*top5_feedback_crop_classification_error_rate(:,4),'m-^'); 


xlabel('$\sigma$','Interpreter','LaTex','FontSize',fontsize);
ylabel('Classification Error (%)','Interpreter','LaTex','FontSize',fontsize);
ylim([0 100])
%legend('show', 'DislpayName', classification_legend(:) ,'Location', 'best');
legend(classification_legend(:),'Location', 'southeast');
set(gca, 'FontSize', fontsize);
set(gcf, 'PaperPosition', [0 0 200 100]);   %Position the plot further to the left and down. Extend the plot to fill entire paper.
set(gcf, 'PaperSize', [200 100]);           %Keep the same paper size
saveas(figure(1),'classification_error_high_blur_caffenet_100.pdf')
%export_fig localization_error_sigma -pdf


%% CLASSIFICATION ERROR PLOTS - VGGNet

classification_legend = {...
    char('top 1 ');...
    char('top 5 ');...
    char('top 1 feedback (fovea) ');...
    char('top 5 feedback (fovea) ');... 
    char('top 1 feedback (crop) ');...
    char('top 5 feedback (crop) ');...
    };


figure(2)
fontsize=20;
set(gcf, 'Color', [1,1,1]);

plot(vgg_sigmas,100*top1_vgg_classification_error_rate(:,3),'k--o'); 
hold on
plot(vgg_sigmas,100*top5_vgg_classification_error_rate(:,3),'k-o'); 

plot(vgg_feedback_sigmas,100*top1_vgg_feedback_classification_error_rate(:,3),'b--s'); 
plot(vgg_feedback_sigmas,100*top5_vgg_feedback_classification_error_rate(:,3),'b-s'); 

plot(vgg_feedback_crop_sigmas,100*top1_vgg_feedback_crop_classification_error_rate(:,3),'m--^'); 
plot(vgg_feedback_crop_sigmas,100*top5_vgg_feedback_crop_classification_error_rate(:,3),'m-^'); 


xlabel('$\sigma$','Interpreter','LaTex','FontSize',fontsize);
ylabel('Classification Error (%)','Interpreter','LaTex','FontSize',fontsize);
ylim([0 100])
%legend('show', 'DislpayName', classification_legend(:) ,'Location', 'best');
legend(classification_legend(:),'Location', 'southeast');
set(gca, 'FontSize', fontsize);
set(gcf, 'PaperPosition', [0 0 200 100]);   %Position the plot further to the left and down. Extend the plot to fill entire paper.
set(gcf, 'PaperSize', [200 100]);           %Keep the same paper size
saveas(figure(2),'classification_error_high_blur_vggnet_100.pdf')
%export_fig localization_error_sigma -pdf


%% CLASSIFICATION ERROR PLOTS - GoogLeNet

classification_legend = {...
    char('top 1 ');...
    char('top 5 ');...
    char('top 1 feedback (fovea) ');...
    char('top 5 feedback (fovea) ');... 
    char('top 1 feedback (crop) ');...
    char('top 5 feedback (crop) ');...
    };


figure(3)
fontsize=20;
set(gcf, 'Color', [1,1,1]);

plot(google_sigmas,100*top1_google_classification_error_rate(:,3),'k--o'); 
hold on
plot(google_sigmas,100*top5_google_classification_error_rate(:,3),'k-o'); 

plot(google_feedback_sigmas,100*top1_google_feedback_classification_error_rate(:,3),'b--s'); 
plot(google_feedback_sigmas,100*top5_google_feedback_classification_error_rate(:,3),'b-s'); 

plot(google_feedback_crop_sigmas,100*top1_google_feedback_crop_classification_error_rate(:,3),'m--^'); 
plot(google_feedback_crop_sigmas,100*top5_google_feedback_crop_classification_error_rate(:,3),'m-^'); 


xlabel('$\sigma$','Interpreter','LaTex','FontSize',fontsize);
ylabel('Classification Error (%)','Interpreter','LaTex','FontSize',fontsize);
ylim([0 100])
%legend('show', 'DislpayName', classification_legend(:) ,'Location', 'best');
legend(classification_legend(:),'Location', 'southeast');
set(gca, 'FontSize', fontsize);
set(gcf, 'PaperPosition', [0 0 200 100]);   %Position the plot further to the left and down. Extend the plot to fill entire paper.
set(gcf, 'PaperSize', [200 100]);           %Keep the same paper size
saveas(figure(3),'classification_error_high_blur_googlenet_100.pdf')
%export_fig localization_error_sigma -pdf

%% LOCALIZATION ERROR PLOTS
% 
% localizaion_legend = {...
%     char('YOLO');...    
%     char('YOLT (fovea)');...
%     char('YOLT (crop)');...
%     char('Local base line');...
%     };
% 
% 
% figure(2)
% fontsize=15;
% set(gcf, 'Color', [1,1,1]);
% 
% plot(sigmas,100*detection_error_rate(:,3),'k-o');           % first pass
% hold on
% plot(sigmas,100*detection_foveate_error_rate(:,3),'b-o');   % fovea
% 
% plot(sigmas,100*detection_crop_error_rate(:,3),'r-o');      % crop
% 
% plot(sigmas,100*repmat(detection_local_error_rate(:,15),length(sigmas)), 'g-o'); % local base line  
% 
% xlabel('$\sigma$','Interpreter','LaTex','FontSize',fontsize);
% ylabel('Localization Error (%)','Interpreter','LaTex','FontSize',fontsize);
% ylim([0 100])
% %legend('show', 'DislpayName', classification_legend(:) ,'Location', 'best');
% legend(localizaion_legend(:),'Location', 'southeast');
% set(gcf, 'PaperPosition', [0 0 200 100]);   %Position the plot further to the left and down. Extend the plot to fill entire paper.
% set(gcf, 'PaperSize', [200 100]);           %Keep the same paper size
% saveas(figure(2),'localization_error_high_blur_caffenet_100.pdf')
% %export_fig localization_error_sigma -pdf


