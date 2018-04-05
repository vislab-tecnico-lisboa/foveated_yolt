clear all
close all

% parameters
detections_resolution=227;
images_number=100; 
overlap_correct=0.5;
top_k=5;

% Images 
images_folder='../data/images/';

% Ground Truth Bounding Boxes and Classes
gt_folder='../data/GroundTruthBBox/';
gt_class_file='../data/ground_truth_labels_ilsvrc12.txt';
[gt_detections, gt_classes]=parse_ground_truth(gt_folder,gt_class_file,images_number);

% Detections from the first feedfoward pass - fix_pts random
detections_file='../results/feedfoward_detection_t21s1p10r1i100.txt';
[sigmas,threshs,fix_pts,classes,scores,detections]=parse_detections(...
    detections_file);
% Detections from the first feedfoward pass - fix_pts (0,0)
detections_file_c='../results/feedfoward_detection_t21s1p1r0i100.txt';
[~,~,fix_pts_c,classes_c,scores_c,detections_c]=parse_detections(...
    detections_file_c);
% Detections from the second feedfoward pass
fb_detections_file='../results/feedback_detection_t21s1p10r1i100.txt';
[~,~,~,rank_feedback_classes,feedback_scores]=feedback_parse_detections(...
    fb_detections_file);
% Detections from the second feedfoward pass - fix_pts (0,0)
fb_detections_file_c='../results/feedback_detection_t21s1p10r1i100.txt';
[~,~,~,rank_feedback_classes_c,feedback_scores_c]=feedback_parse_detections(...
    fb_detections_file_c);

%% Classification Errors
% [~, ~, class_error_5av, class_error_5av_std] =...
%     classification_error_rates(...
%     sigmas,threshs,fix_pts,images_number,classes,gt_classes,top_k);
% 
% [~, ~, rank_fbclass_error_5_av,rank_fbclass_error_5_std] =...
%     classification_error_rates(...
%     sigmas,threshs,fix_pts,images_number,rank_feedback_classes,gt_classes,top_k);
% 
% [~, ~, class_error_5av_c, class_error_5av_std_c] =...
%     classification_error_rates(...
%     sigmas,threshs,fix_pts_c,images_number,classes_c,gt_classes,top_k);
% 
% [~, ~, rank_fbclass_error_5_av_c,rank_fbclass_error_5_std_c] =...
%     classification_error_rates(...
%     sigmas,threshs,fix_pts_c,images_number,rank_feedback_classes_c,gt_classes,top_k);


%% Localization Errors

[local_error_av,local_error_std] = detection_error_rates(...
    sigmas,threshs,fix_pts,images_number,detections,gt_detections,detections_resolution,top_k,overlap_correct);
[local_error_av_c,local_error_std_c] = detection_error_rates(...
    sigmas,threshs,fix_pts_c,images_number,detections_c,gt_detections,detections_resolution,top_k,overlap_correct);

%%
% figure(1)
% errorbar(sigmas,100* class_error_5av, 100*class_error_5av_std,'k-');
% hold on
% plot(sigmas,100* class_error_5av_c,'b-');
% hold on
% errorbar(sigmas, 100*rank_fbclass_error_5_av,100*rank_fbclass_error_5_std,'r-');
% hold on
% plot(sigmas, 100*rank_fbclass_error_5_av_c,'g-');
% title('Top 5 - Central Vs Random Initial Fixation Point');
% ylim([0 1.2*100]);
% xlabel('\sigma');
% ylabel('Classification Error (%)');
% legend('1st pass 10 rand fixpts','1st pass (0,0)','2nd pass 10 rand fixpts','2nd pass (0,0)');

%%
figure(2)
errorbar(threshs,100*local_error_av,100*local_error_std);
hold on
plot(threshs,100*local_error_av_c);
title('Random Initial Fixation Point');
xlabel('\sigma');
ylabel('Localization Error (%)');
legend('10 random','(0,0)');
ylim([0 1.2*100]);



