clear all

% parameters
detections_resolution=227;
images_number=100; 
overlap_correct=0.5;
top_k=5;
plots=true;

% Images 
images_folder='../data/images/';

% Ground Truth Bounding Boxes and Classes
gt_folder='../data/GroundTruthBBox/';
gt_class_file='../data/ground_truth_labels_ilsvrc12.txt';
[gt_detections, gt_classes]=parse_ground_truth(gt_folder,gt_class_file,images_number);
%%
% baseline

% File for Classification
baseline_file='../results/google_baselinefeedfoward_detection_t1s1p1r0i100.txt';
[b_sigmas,b_thresh, b_fix_pts, b_classes, b_scores, b_detections]=parse_detections(...
    baseline_file);
%%
% Classification Error
[~, ~, C_baseline, ~,~,~]=classification_error_rates(...
    b_sigmas,b_thresh,b_fix_pts,images_number,b_classes,gt_classes,top_k);
[~, L_baseline, ~]=detection_error_rates(...
    b_sigmas,b_thresh,b_fix_pts,images_number,b_detections,gt_detections,detections_resolution,top_k,overlap_correct);


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%% CENTRAL FIXATION POINT - (112,112) %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Classification Erros Vs f0
C_detections_file0_s='../results/google_class0feedfoward_detection_t1s20p1r0i100.txt';
C_fb_detections_file0_s='../results/google_class0feedback_detection_t1s20p1r0i100.txt';

[C_sigmas0,~,~,~,C_error5_av0_s,~,~,C_fb_error5_av0_s,~]=CErrors(...
    images_number,top_k,gt_classes,C_detections_file0_s,C_fb_detections_file0_s);
C_gain0_s = (C_fb_error5_av0_s'- C_error5_av0_s')./C_fb_error5_av0_s';

%% Classification Erros Vs th
C_detections_file0_t='../results/google_loc0feedfoward_detection_t21s1p1r0i100.txt';
C_fb_detections_file0_t='../results/google_loc0feedback_detection_t21s1p1r0i100.txt';

[~,C_threshs0,~,~,C_error5_av0_t,~,~,C_fb_error5_av0_t,~]=CErrors(...
    images_number,top_k,gt_classes,C_detections_file0_t,C_fb_detections_file0_t);
C_gain0_t = (C_fb_error5_av0_t'- C_error5_av0_t')./C_fb_error5_av0_t';

%% Localization Erros Vs f0
L_detections_file0_s='../results/google_class0feedfoward_detection_t1s20p1r0i100.txt';
L_fb_detections_file0_s='../results/google_class0feedback_detection_t1s20p1r0i100.txt';

[~,L_sigmas0,~,~,L_error_av0_s,~,~,L_fb_error_av0_s,~]=LErrors(...
    images_number,detections_resolution,overlap_correct,top_k,gt_detections,L_detections_file0_s,L_fb_detections_file0_s);

L_gain0_s = (L_fb_error_av0_s' - L_error_av0_s')./L_fb_error_av0_s';

%% Localization Erros Vs th
L_detections_file0_t='../results/google_loc0feedfoward_detection_t21s1p1r0i100.txt';
L_fb_detections_file0_t='../results/google_loc0feedback_detection_t21s1p1r0i100.txt';

[L_threshs0,~,~,~,L_error_av0_t,~,~,L_fb_error_av0_t,~]=LErrors(...
    images_number,detections_resolution,overlap_correct,top_k,gt_detections,L_detections_file0_t,L_fb_detections_file0_t);

L_gain0_t = (L_fb_error_av0_t' - L_error_av0_t')./L_fb_error_av0_t';


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%% 16 Points       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Classification Erros Vs f0
C_detections_file16_s='../results/google_classfeedfoward_detection_t1s20p16r0i100.txt';
C_fb_detections_file16_s='../results/google_classfeedback_detection_t1s20p16r0i100.txt';

[C_sigmas,~,~,~,C_error5_av_s,C_error5_std_s,~,C_fb_error5_av_s,C_fb_error5_std_s]=CErrors(...
    images_number,top_k,gt_classes,C_detections_file16_s,C_fb_detections_file16_s);
C_gain_s = (abs(C_fb_error5_av_s'- C_error5_av_s'))./C_error5_av_s';

%% Classification Erros Vs th
C_detections_file16_t='../results/google_locfeedfoward_detection_t21s1p16r0i100.txt';
C_fb_detections_file16_t='../results/google_locfeedback_detection_t21s1p16r0i100.txt';

[~,C_threshs,~,~,C_error5_av_t,C_error5_std_t,~,C_fb_error5_av_t,C_fb_error5_std_t]=CErrors(...
    images_number,top_k,gt_classes,C_detections_file16_t,C_fb_detections_file16_t);
C_gain_t = (abs(C_fb_error5_av_t'- C_error5_av_t'))./C_fb_error5_av_t';

%% Localization Erros Vs f0
L_detections_file16_s='../results/google_classfeedfoward_detection_t1s20p4r0i100.txt';
L_fb_detections_file16_s='../results/google_classfeedback_detection_t1s20p4r0i100.txt';

[~,L_sigmas,~,~,L_error_av_s,L_error_std_s,~,L_fb_error_av_s,L_fb_error_std_s]=LErrors(...
    images_number,detections_resolution,overlap_correct,top_k,gt_detections,L_detections_file16_s,L_fb_detections_file16_s);

L_gain_s = (abs(L_fb_error_av_s' - L_error_av_s'))./L_fb_error_av_s';

%% Localization Errors Vs th
L_detections_file16_t='../results/google_locfeedfoward_detection_t21s1p16r0i100.txt';
L_fb_detections_file16_t='../results/google_locfeedback_detection_t21s1p16r0i100.txt';

[L_threshs,~,~,~,L_error_av_t,L_error_std_t,~,L_fb_error_av_t,L_fb_error_std_t]=LErrors(...
    images_number,detections_resolution,overlap_correct,top_k,gt_detections,C_detections_file16_t,C_fb_detections_file16_t);

L_gain_t = (abs(L_fb_error_av_t' - L_error_av_t'))./L_fb_error_av_t';

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%% 64 Points       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

C_detections_file64='../results/google_pos_classfeedfoward_detection_t1s10p64r0i100.txt';
C_fbdetections_file64='../results/google_pos_classfeedback_detection_t1s10p64r0i100.txt';
[~,C_fix_pts,C_error_pos,~,~,C_fb_error_pos,~,~]=CErrors(...
   images_number,top_k,gt_classes,C_detections_file64,C_fbdetections_file64);

%%
L_detections_file64='../results/google_pos_locfeedfoward_detection_t6s1p64r0i100.txt';
L_fb_detections_file64='../results/google_pos_locfeedback_detection_t6s1p64r0i100.txt';

[~,~,L_fix_pts,L_error_pos,~,~,L_fb_error_pos,~,~]=LErrors(...
    images_number,detections_resolution,overlap_correct,top_k,gt_detections,L_detections_file64,L_fb_detections_file64);


