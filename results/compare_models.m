close all
addpath('export_fig');

gt_folder='../dataset/gt/';

detections_caffe_file='../dataset/detections/new/raw_bbox_parse_crop_high_blur_caffenet_100.txt';   
detections_vgg_file='../dataset/detections/new/raw_bbox_parse_crop_high_blur_vggnet_100.txt';

detections_file_base_localization='../dataset/detections/new/raw_bbox_parse_local_yolo_high_resolution_caffenet_1000.txt';

% Feedback
% CaffeNet
feedback_detections_caffe_file = '../dataset/detections/new/feedback_detection_parse_fovea_high_blur_caffenet_100.txt';
feedback_crop_detections_caffe_file = '../dataset/detections/new/feedback_detection_parse_crop_high_blur_caffenet_100.txt';

% VGGNet
%feedback_detections_vgg_file = '../dataset/detections/new/feedback_detection_parse_fovea_high_blur_vggnet_100.txt';
feedback_crop_vgg_detections_file = '../dataset/detections/new/feedback_detection_parse_crop_high_blur_vggnet_100.txt';


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
    detections_caffe_file);

% get detections (YOLO) - VGG
[vgg_sigmas,vgg_threshs,vgg_classes,vgg_scores,vgg_detections]=parse_detections2(...
    images_number,...
    detections_vgg_file);

% get detections (best localization base line)
[local_sigmas,local_threshs,local_classes,local_scores,local_detections]=parse_detections2(...
    images_number,...
    detections_file_base_localization);

% CaffeNet
% get feedback detections (Fovea) 
[feedback_sigmas,feedback_threshs,feedback_classes,feedback_scores,rank_feedback_classes,feedback_detections]=feedback_parse_detections2(...
    images_number,...
    feedback_detections_caffe_file);

% get feedback detections (Crop) 
[feedback_crop_sigmas,feedback_crop_threshs,feedback_crop_classes,feedback_crop_scores,rank_feedback_crop_classes,feedback_crop_detections]=feedback_parse_detections2(...
    images_number,...
    feedback_crop_detections_caffe_file);


% VGGNet
% get feedback detections (Fovea) 
% [feedback_vgg_sigmas,feedback_vgg_threshs,feedback_vgg_classes,feedback_vgg_scores,rank_feedback_vgg_classes,feedback_vgg_detections]=feedback_parse_detections2(...
%     images_number,...
%     feedback_detections_vgg_file);

% get feedback detections (Crop) 
[feedback_crop_vgg_sigmas,feedback_crop_vgg_threshs,feedback_crop_vgg_classes,feedback_crop_vgg_scores,rank_feedback_crop_vgg_classes,feedback_crop_vgg_detections]=feedback_parse_detections2(...
    images_number,...
    feedback_crop_vgg_detections_file);

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
% CAFFENET
% get classification error rates (YOLO)
[top1_classification_error_rate, top5_classification_error_rate] = classification_error_rates(sigmas,threshs,images_number,classes,gt_classes,top_k);

% get feedback classification error rates (Fovea)
[top1_feedback_classification_error_rate, top5_feedback_classification_error_rate] = classification_error_rates(feedback_sigmas,feedback_threshs,images_number,rank_feedback_classes,gt_classes,top_k);

% get feedback classification error rates (crop)
[top1_feedback_crop_classification_error_rate, top5_feedback_crop_classification_error_rate] = classification_error_rates(feedback_crop_sigmas,feedback_crop_threshs,images_number,rank_feedback_crop_classes,gt_classes,top_k);


% VGGNET
% get classification error rates (YOLO)
[top1_vgg_classification_error_rate, top5_vgg_classification_error_rate] = classification_error_rates(vgg_sigmas,vgg_threshs,images_number,vgg_classes,gt_classes,top_k);

% get feedback classification error rates (Fovea)
%[top1_vgg_feedback_classification_error_rate, top5_vgg_feedback_classification_error_rate] = classification_error_rates(feedback_vgg_sigmas,feedback_vgg_threshs,images_number,rank_feedback_vgg_classes,gt_classes,top_k);

% get feedback classification error rates (crop)
[top1_vgg_feedback_crop_classification_error_rate, top5_vgg_feedback_crop_classification_error_rate] = classification_error_rates(feedback_crop_vgg_sigmas,feedback_crop_vgg_threshs,images_number,rank_feedback_crop_vgg_classes,gt_classes,top_k);



%% CLASSIFICATION ERROR PLOTS - Caffe vs Google vs Vgg

classification_legend = {...
    char('top 1 feedback (fovea) CaffeNet');...
    char('top 5 feedback (fovea) CaffeNet');... 
    char('top 1 feedback (crop)  CaffeNet');...
    char('top 5 feedback (crop)  CaffeNet');...
    %char('top 1 feedback (fovea) VGGNet');...
    %char('top 5 feedback (fovea) VGGNet');... 
    char('top 1 feedback (crop)  VGGNet');...
    char('top 5 feedback (crop)  VGGNet');...
    };


figure(1)
fontsize=15;
set(gcf, 'Color', [1,1,1]);
aux=feedback_crop_vgg_sigmas;
aux = aux(1:11,:);
%plot(sigmas,100*top1_classification_error_rate(:,2),'k-*'); 
%hold on
%plot(sigmas,100*top5_classification_error_rate(:,2),'k-o'); 

plot(feedback_sigmas,100*top1_feedback_classification_error_rate(:,3),'b--*'); 
hold on
plot(feedback_sigmas,100*top5_feedback_classification_error_rate(:,3),'b-*'); 

plot(feedback_crop_sigmas,100*top1_feedback_crop_classification_error_rate(:,4),'c--o'); 
plot(feedback_crop_sigmas,100*top5_feedback_crop_classification_error_rate(:,4),'c-o'); 

%plot(feedback_vgg_sigmas,100*top1_vgg_feedback_classification_error_rate(:,3),'r--^'); 
%plot(feedback_vgg_sigmas,100*top5_vgg_feedback_classification_error_rate(:,3),'r-^'); 

plot(aux,100*repmat(top1_vgg_feedback_crop_classification_error_rate(:,4),length(aux)),'m--s'); 
plot(aux,100*repmat(top5_vgg_feedback_crop_classification_error_rate(:,4),length(aux)),'m-s'); 


xlabel('$\sigma$','Interpreter','LaTex','FontSize',fontsize);
ylabel('Classification Error (%)','Interpreter','LaTex','FontSize',fontsize);
ylim([0 100])
%legend('show', 'DislpayName', classification_legend(:) ,'Location', 'best');
legend(classification_legend(:),'Location', 'southeast');
set(gcf, 'PaperPosition', [0 0 200 100]);   %Position the plot further to the left and down. Extend the plot to fill entire paper.
set(gcf, 'PaperSize', [200 100]);           %Keep the same paper size
saveas(figure(1),'classification_error_high_blur_caffe_vs_vgg_100.pdf')
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


