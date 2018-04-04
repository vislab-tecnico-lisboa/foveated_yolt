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

% Detections from the first feedfoward pass
detections_file='../results/yolt_exp2_random/feedforward_detection_parse.txt';

% Parse
[sigmas,threshs,fix_pts,classes,scores,detections]=parse_detections(...
    detections_file);

% Detections from the second feedfoward pass
fb_detections_file='../results/yolt_exp2_random/feedback_detection_parse.txt';

% Parse
[~,~,~,rank_feedback_classes,feedback_scores]=feedback_parse_detections(...
    fb_detections_file);

%% Classification Errors
[class_error_1_av, class_error_1_std, class_error_5av, class_error_5av_std] =...
    classification_error_rates(...
    sigmas,threshs,fix_pts,images_number,classes,gt_classes,top_k);

[rank_fbclass_error_1_av, rank_fbclass_error_1_std, rank_fbclass_error_5_av,rank_fbclass_error_5_std] =...
    classification_error_rates(...
    sigmas,threshs,fix_pts,images_number,rank_feedback_classes,gt_classes,top_k);

%% Localization Errors

[local_error_av,local_error_std] = detection_error_rates(...
    sigmas,threshs,fix_pts,images_number,detections,gt_detections,detections_resolution,top_k,overlap_correct);

%%
figure(1)
errorbar(sigmas,100* class_error_5av, 100*class_error_5av_std,'b-');
hold on
errorbar(sigmas, 100*rank_fbclass_error_5_av,100*rank_fbclass_error_5_std,'r-');
title('Random Initial Fixation Point');
ylim([0 1.2*100]);
xlabel('\sigma');
ylabel('Classification Error (%)');
legend('Top 5 1st pass','Top 5 2nd pass ranked');

%%
figure(2)
errorbar(sigmas,100*local_error_av,100*local_error_std);
title('Random Initial Fixation Point');
xlabel('\sigma');
ylabel('Localization Error (%)');
legend('th=0.65');
ylim([0 1.2*100]);



