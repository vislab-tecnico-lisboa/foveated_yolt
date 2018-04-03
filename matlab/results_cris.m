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
detections_file='../results/test_ana_center_localization_errors/feedforward_detection_parse.txt';

% Parse
[sigmas,threshs,classes,scores,detections]=parse_detections(...
    images_number,...
    detections_file);

% Detections from the second feedfoward pass
fb_detections_file='../results/test_ana_center_localization_errors/feedback_detection_parse.txt';

% Parse
[feedback_sigmas,feedback_threshs,rank_feedback_classes,feedback_classes,feedback_scores]=feedback_parse_detections(...
    images_number,...
    fb_detections_file);

%% Classification Errors
[class_error_1, class_error_5] = classification_error_rates(...
    sigmas,threshs,images_number,classes,gt_classes,top_k);

[rank_fbclass_error_1, rank_fbclass_error_5] = classification_error_rates(...
    feedback_sigmas,feedback_threshs,images_number,rank_feedback_classes,gt_classes,top_k);

[fbclass_error_1, fbclass_error_5] = classification_error_rates(...
    feedback_sigmas,feedback_threshs,images_number,feedback_classes,gt_classes,top_k);

%% Localization Errors

[local_error] = detection_error_rates(...
    sigmas,threshs,images_number,detections,gt_detections,detections_resolution,top_k,overlap_correct);

%%
figure(1)
plot(sigmas, class_error_5,'b-')
hold on
plot(sigmas, rank_fbclass_error_5,'r-')
hold on
%plot(sigmas, rank_fbclass_error_5,'g-')
%hold on
plot(sigmas, class_error_1,'b--')
hold on
plot(sigmas, rank_fbclass_error_1,'r--')
title('Centered Initial Fixation Point');
xlabel('\sigma');
ylabel('Classification Error (%)');
legend('Top 5 1st pass','Top 5 2nd pass ranked','Top 1 1st pass','Top 1 2nd pass ranked')
%legend('Top 5 1st pass','Top 5 2nd pass', 'Top 5 2nd pass ranked');
ylim([0 1.2]);

%%
figure(2)
plot(threshs,local_error);
title('Centered Initial Fixation Point');
xlabel('Th');
ylabel('Localization Error');
legend('\sigma=70');
ylim([0 1.2]);
