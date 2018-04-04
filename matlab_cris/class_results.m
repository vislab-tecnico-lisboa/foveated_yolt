clear all
close all

% parameters
detections_resolution=227;
images_number=10; 
overlap_correct=0.5;
top_k=5;
plots = true;

% Images 
images_folder='../data/images/';

% Ground Truth Bounding Boxes and Classes
gt_folder='../data/GroundTruthBBox/';
gt_class_file='../data/ground_truth_labels_ilsvrc12.txt';

[gt_detections, gt_classes]=parse_ground_truth(gt_folder,gt_class_file,images_number);

% Detections from the first feedfoward pass
detections_file='../resuts/test/feedforward_detection_parse.txt';
detection_file_c='../resuts/test_center/feedforward_detection_parse.txt';
[sigmas,threshs,fix_pts,classes,scores,detections] = parse_detections(detections_file);
[~,threshs_c,fix_pts_c,classes_c,scores_c,detections_c] = parse_detections(detection_file_c);

% Detections from the first feedfoward pass
fb_detections_file='../resuts/test/feedback_detection_parse.txt';
fb_detections_file_c='../resuts/test_center/feedback_detection_parse.txt'; 

[~,~,~,fb_classes,fb_scores,fb_detections] = parse_detections(fb_detections_file);
[~,~,~,fb_classes_c,fb_scores_c,fb_detections_c] = parse_detections(fb_detections_file_c);

% Classification Error

[tp1_class_error_av,tp1_class_error_std,tp5_class_error_av, tp5_class_error_std] = classification_error_rates(sigmas,threshs,fix_pts,images_number,classes,gt_classes,top_k);
[tp1_class_error_av_c,tp1_class_error_std_c,tp5_class_error_av_c, tp5_class_error_std_c] = classification_error_rates(sigmas,threshs_c,fix_pts_c,images_number,classes_c,gt_classes,top_k);

[fb_tp1_class_error_av,fb_tp1_class_error_std,fb_tp5_class_error_av,fb_tp5_class_error_std] = classification_error_rates(sigmas,threshs,fix_pts,images_number,fb_classes,gt_classes,top_k);
[fb_tp1_class_error_av_c,fb_tp1_class_error_std_c,fb_tp5_class_error_av_c,fb_tp5_class_error_std_c] = classification_error_rates(sigmas,threshs_c,fix_pts_c,images_number,fb_classes_c,gt_classes,top_k);

% Classification Plots


figure(1)
plot(sigmas, tp5_class_error_av_c,'b-')
hold on
plot(sigmas, fb_tp5_class_error_av_c,'r-')
hold on
plot(sigmas, tp1_class_error_av_c,'b--')
hold on
plot(sigmas, fb_tp1_class_error_av_c,'r--')
    title('Centered Initial Fixation Point');
    xlabel('\sigma');
    ylabel('Classification Error (%)');
    ylim([0 1.2]);

if plots
    figure(1)
    errorbar(sigmas,tp5_class_error_av(4,:),tp5_class_error_std(4,:),'b-o'); 
    hold on
    errorbar(sigmas,tp1_class_error_av(4,:),tp1_class_error_std(4,:),'b--o'); 
    hold on    
    errorbar(sigmas,fb_tp5_class_error_av(4,:),fb_tp5_class_error_std(4,:),'r-o'); 
    hold on 
    errorbar(sigmas,fb_tp1_class_error_av(4,:),fb_tp1_class_error_std(4,:),'r--o');
    title('Spread Initial Fixation Points');
    xlabel('\sigma');
    ylabel('Classification Error (%)');
    ylim([0 1.2]);
    legend('Top5-1st pass','Top1-1st pass','Top5-2nd pass','Top1-2nd pass')
end
%%



if plots
    figure(2)
%    errorbar(sigmas,tp5_class_error_av(4,:),tp5_class_error_std(4,:),'r-o'); 
%    hold on 
%    errorbar(sigmas,tp1_class_error_av(4,:),tp1_class_error_std(4,:),'r--o');
%    errorbar(sigmas,tp5_class_error_av_c,tp5_class_error_std_c,'b-o'); 
%    hold on
%    errorbar(sigmas,tp1_class_error_av_c,tp1_class_error_std_c,'b--o'); 
%    hold on    
    errorbar(sigmas,tp5_class_error_av_c,tp5_class_error_std_c,'b-o'); 
    hold on 
    errorbar(sigmas,tp1_class_error_av_c,tp1_class_error_std_c,'b--o');
    title('Centered Initial Fixation Point');
    xlabel('\sigma');
    ylabel('Classification Error (%)');
    ylim([0 1.2]);
%    legend('Top5-2nd pass','Top1-2nd pass','Top5-2nd pass center','Top1-2nd pass center')
end


% Localization Error

%[error_rate] = detection_error_rates(sigmas,threshs,fix_pts,images_number,detections,gt_detections,detections_resolution,top_k,overlap_correct);
%[fb_error_rate] = detection_error_rates(fb_sigmas,fb_threshs,fb_fix_pts,images_number,fb_detections,gt_detections,detections_resolution,top_k,overlap_correct);


% if plots
%    figure(2)
%     plot(threshs,100*error_rate(1),'-o'); 
%     hold on
%     plot(fb_threshs,100*fb_error_rate(1),'r-o'); 
%     xlabel('th');
%     ylabel('Localization Error (%)');
%     ylim([0 100]);
%     legend('First pass only','Second pass')
% end


