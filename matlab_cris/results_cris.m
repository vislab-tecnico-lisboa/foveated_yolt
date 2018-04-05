clear all
load('parameters.mat');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%% 10 RANDOM FIXATION POINTS %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load('files_10random.mat');

% %% Files for Classification Errors
% % Detections from the 1st pass - sigmas
% C_detections_file='../results/feedfoward_detection_t1s16p10r1i100.txt';
% [C_sigmas, C_threshs, C_fix_pts, C_classes, C_scores, C_detections]=parse_detections(...
%     C_detections_file);
% % Detections from the 2nd pass - sigmas
% C_fbdetections_file='../results/feedback_detection_t1s16p10r1i100.txt';
% [~,~,~,C_fb_rankclasses,C_fb_classes,C_fb_scores]=feedback_parse_detections(...
%     C_fbdetections_file);
% 
% %% Files for Localization Errors
% % Detections from the 1st pass - thresh
% L_detections_file='../results/feedfoward_detection_t21s1p10r1i100.txt';
% [L_sigmas, L_threshs, L_fix_pts, L_classes, L_scores, L_detections]=parse_detections(...
%     L_detections_file);

%% Classification Error
% 1st pass
[~, ~, C_error5_av, C_error5_std]=classification_error_rates(...
    C_sigmas,C_threshs,C_fix_pts,images_number,C_classes,gt_classes,top_k);
% 2nd pass Top 5 - Ranked with != labels
[~, ~, C_fb_error5_r_av,C_fb_error5_r_std]=classification_error_rates(...
    C_sigmas,C_threshs,C_fix_pts,images_number,C_fb_rankclasses,gt_classes,top_k);
% 2nd pass Top 5 - could have same labels
[~, ~, C_fb_error5_av,C_fb_error5_std]=classification_error_rates(...
    C_sigmas,C_threshs,C_fix_pts,images_number,C_fb_classes,gt_classes,top_k);

%% Localization Error
[L_error_av, L_error_std] = detection_error_rates(...
    L_sigmas,L_threshs,L_fix_pts,images_number,L_detections,...
    gt_detections,detections_resolution,top_k,overlap_correct);

%% Plots
figure(1)
errorbar(C_sigmas,100*C_error5_av, 100*C_error5_std,'b-');
hold on
plot(C_sigmas,100* C_error5_av0,'b--');
hold on
errorbar(C_sigmas, 100*C_fb_error5_r_av,100*C_fb_error5_r_std,'r-');
hold on
plot(C_sigmas, 100*C_fb_error5_av0,'r--');
title('Classification Errors - Central Vs 10 Random Initial Fixation Points');
ylim([0 1.2*100]);
xlabel('\sigma');
ylabel('Classification Error (%)');
legend('1st pass-10 rand fixpts','1st pass-(0,0)','2nd pass-10 rand fixpts','2nd pass-(0,0)');

figure(2)
errorbar(C_sigmas,100*C_error5_av, 100*C_error5_std,'b-');
hold on
errorbar(C_sigmas,100*C_fb_error5_r_av, 100*C_fb_error5_r_std,'r-');
hold on
errorbar(C_sigmas,100*C_fb_error5_av, 100*C_fb_error5_std,'g-');
title({'Classification Errors - Ranked with diff labels or same labels';'(10 Random Initial Fixation Points)'});
ylim([0 1.2*100]);
xlabel('\sigma');
ylabel('Classification Error (%)');
legend('1st pass','2nd pass-top 5 ranked w diff labels','2nd pass-top 5');


figure(3)
errorbar(L_threshs,100*L_error_av,100*L_error_std, 'b-');
hold on
plot(L_threshs,100*L_error_av0,'r-');
title('Localization Errors - Central Vs 10 Random Initial Fixation Points');
xlabel('Th');
ylabel('Localization Error (%)');
legend('10 random fixation points','1 central fixation point');
ylim([0 1.2*100]);



