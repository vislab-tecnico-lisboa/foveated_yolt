clear all

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
run centralpoint.m

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%% CENTRAL VS 16 SPREAD FIXATION POINTS %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Files for Classification Errors
% Detections from the first feedfoward pass - fix_pts random
C_detections_file='../results/feedfoward_detection_t1s16p16r0i100.txt';
[C_sigmas, C_threshs, C_fix_pts, C_classes, C_scores, C_detections]=parse_detections(...
    C_detections_file);
% Detections from the second feedfoward pass - fix_pts random
C_fbdetections_file='../results/feedback_detection_t1s16p16r0i100.txt';
[~,~,~,C_fb_rankclasses,C_fb_classes,C_fb_scores]=feedback_parse_detections(...
    C_fbdetections_file);

%% Files for Localization Errors
Detections from the 1st pass - thresh
L_detections_file='../results/feedfoward_detection_t21s1p16r0i100.txt';
[L_sigmas, L_threshs, L_fix_pts, L_classes, L_scores, L_detections]=parse_detections(...
    L_detections_file);

%% Classification Error
% 1st pass
[~, ~, C_error5_av, C_error5_std,~,C_error5_pos]=classification_error_rates(...
    C_sigmas,C_threshs,C_fix_pts,images_number,C_classes,gt_classes,top_k);
% 2nd pass Top 5 - Ranked with != labels
[~, ~, C_fb_error5_r_av,C_fb_error5_r_std,~,C_fb_error5_r_pos]=classification_error_rates(...
    C_sigmas,C_threshs,C_fix_pts,images_number,C_fb_rankclasses,gt_classes,top_k);
% 2nd pass Top 5 - could have same labels
[~, ~, C_fb_error5_av,C_fb_error5_std,~,~]=classification_error_rates(...
    C_sigmas,C_threshs,C_fix_pts,images_number,C_fb_classes,gt_classes,top_k);


%% Localization Error
[L_error_av, L_error_std] = detection_error_rates(...
    L_sigmas,L_threshs,L_fix_pts,images_number,L_detections,...
    gt_detections,detections_resolution,top_k,overlap_correct);

%% Plots
figure(3)
errorbar(C_sigmas,100*C_error5_av, 100*C_error5_std,'b-');
hold on
plot(C_sigmas,100* C_error5_av0,'b--');
hold on
errorbar(C_sigmas, 100*C_fb_error5_r_av,100*C_fb_error5_r_std,'r-');
hold on
plot(C_sigmas, 100*C_fb_error5_av0,'r--');
title({'Classification Errors';'Central Vs 16 Spread Initial Fixation Points'});
ylim([0 1.2*100]);
xlabel('\sigma');
ylabel('Classification Error (%)');
legend('1st pass-4\times4 fixpts','1st pass-(0,0)','2nd pass-4\times4 fixpts','2nd pass-(0,0)');

figure(4)
errorbar(C_sigmas,100*C_error5_av, 100*C_error5_std,'b-');
hold on
errorbar(C_sigmas,100*C_fb_error5_r_av, 100*C_fb_error5_r_std,'r-');
hold on
errorbar(C_sigmas,100*C_fb_error5_av, 100*C_fb_error5_std,'g-');
title({'Classification Errors';'Ranked with diff labels Vs same labels';'(16 Spread Initial Fixation Points)'});
ylim([0 1.2*100]);
xlabel('\sigma');
ylabel('Classification Error (%)');
legend('1st pass','2nd pass-top 5 ranked w diff labels','2nd pass-top 5');

figure(5)
data = [C_error5_pos(:,1:4);C_error5_pos(:,5:8);C_error5_pos(:,9:12);C_error5_pos(:,13:16)];
heatmap(C_fix_pts(1:4),C_fix_pts(1:4),data,'Colormap',bone,'XLabel',...
    'x','YLabel','y','Title','Classification Error with Position - 1st pass',...
    'ColorLimits',[0.5 0.7]);

figure(6)
data = [C_fb_error5_r_pos(:,1:4);C_fb_error5_r_pos(:,5:8);C_fb_error5_r_pos(:,9:12);C_fb_error5_r_pos(:,13:16)];
heatmap(C_fix_pts(1:4),C_fix_pts(1:4),data,'Colormap',bone,'XLabel',...
    'x','YLabel','y','Title','Classification Error with Position - 2nd pass',...
    'ColorLimits',[0.5 0.7]);

figure(7)
errorbar(L_threshs,100*L_error_av,100*L_error_std, 'b-');
hold on
plot(L_threshs,100*L_error_av0,'r-');
title('Localization Errors - Central Vs 16 Spread Initial Fixation Points');
xlabel('Th');
ylabel('Localization Error (%)');
legend('16 random fixation points','1 central fixation point');
ylim([0 1.2*100]);
