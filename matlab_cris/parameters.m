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


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%% CENTRAL FIXATION POINT - (113,113) %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Files 
% Detections from the 1st pass - sigmas
C_detections_file0='../results/feedfoward_detection_t1s16p1r0i100.txt';
[C_sigmas,C_threshs, fix_pts0, C_classes0, C_scores0, C_detections0]=parse_detections(...
    C_detections_file0);
% Detections from the 2nd pass - sigmas 
C_fbdetections_file0='../results/feedback_detection_t1s16p1r0i100.txt';
[~,~,~,C_fb_classes0,C_fb_scores0]=feedback_parse_detections(...
    C_fbdetections_file0);
% Detections from the 1st pass - thresh
L_detections_file0='../results/feedfoward_detection_t21s1p1r0i100.txt';
[L_sigmas,L_threshs, fix_pts0, L_classes0, L_scores0, L_detections0]=parse_detections(...
    L_detections_file0);
% Detections from the 2nd pass - thresh
L_fbdetections_file0='../results/feedback_detection_t21s1p1r0i100.txt';
[~,~,~,L_fb_classes0,L_fb_scores0]=feedback_parse_detections(...
    L_fbdetections_file0);

%% Classification Error
[~, ~, C_error5_av0, C_error5_std0]=classification_error_rates(...
    C_sigmas,C_threshs,fix_pts0,images_number,C_classes0,gt_classes,top_k);
[~, ~, C_fb_error5_av0,C_fb_error5_std0]=classification_error_rates(...
    C_sigmas,C_threshs,fix_pts0,images_number,C_fb_classes0,gt_classes,top_k);

%% Localization Error
[L_error_av0,L_error_std0] = detection_error_rates(...
    L_sigmas,L_threshs,fix_pts0,images_number,L_detections0,...
    gt_detections,detections_resolution,top_k,overlap_correct);

