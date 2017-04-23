close all
addpath('export_fig');

gt_folder='../dataset/gt/';

classifications_file='../files/ground_truth_labels_ilsvrc12.txt';
images_folder='../dataset/images/';

% set to true to check detections
view_detections=false;

% parameters
detections_resolution=227;
images_number=100; 
overlap_correct=0.5;
top_k=5;


%% Load data

% Caffe
foveal_detection ='../dataset/detections/new/raw_bbox_parse_foveal_caffe2.txt';
foveal2_detection ='../dataset/detections/new/feedback_detection_foveal_caffe2.txt';

% Google
google_foveal_detection ='../dataset/detections/new/raw_bbox_parse_foveal_google2.txt';
google_foveal2_detection ='../dataset/detections/new/feedback_detection_foveal_google2.txt';

% VGG
vgg_foveal_detection ='../dataset/detections/new/raw_bbox_parse_foveal_vgg.txt';
vgg_foveal2_detection ='../dataset/detections/new/feedback_detection_foveal_vgg.txt';



%% DETECTIONS - First passage

% get ground truth
[gt_detections, gt_classes]=parse_ground_truth(gt_folder,classifications_file,images_number);

% get detections (YOLO) - FOVEAL - CAFFE
[foveal_sigmas,foveal_threshs,foveal_classes,foveal_scores,foveal_detections]=parse_detections(...
    images_number,...
    foveal_detection);

% get detections (YOLO) - FOVEAL - GOOGLE
[google_foveal_sigmas,google_foveal_threshs,google_foveal_classes,google_foveal_scores,google_foveal_detections]=parse_detections(...
    images_number,...
    google_foveal_detection);

% get detections (YOLO) - FOVEAL - VGG
[vgg_foveal_sigmas,vgg_foveal_threshs,vgg_foveal_classes,vgg_foveal_scores,vgg_foveal_detections]=parse_detections(...
    images_number,...
    vgg_foveal_detection);


%% DETECTIONS - Second passage

% get detections (YOLT) - FOVEAL - CAFFE 
[feedback_sigmas,feedback_threshs,feedback_classes,feedback_scores,rank_feedback_classes,feedback_detections]=feedback_parse_detections2(...
    images_number,...
    foveal2_detection);

% get detections (YOLT) - FOVEAL - GOOGLE 
[google_feedback_sigmas,google_feedback_threshs,google_feedback_classes,google_feedback_scores,google_rank_feedback_classes,google_feedback_detections]=feedback_parse_detections2(...
    images_number,...
    google_foveal2_detection);

% get detections (YOLT) - FOVEAL - VGG 
[vgg_feedback_sigmas,vgg_feedback_threshs,vgg_feedback_classes,vgg_feedback_scores,vgg_rank_feedback_classes,vgg_feedback_detections]=feedback_parse_detections2(...
    images_number,...
    vgg_foveal2_detection);


%% LOCALIZATION

% First passage
% get detection error rates (YOLO) - FOVEAL - CAFFE
[foveal_detection_error_rate] = detection_error_rates(foveal_sigmas,foveal_threshs,images_number,foveal_detections,gt_detections,detections_resolution,top_k,overlap_correct);

% get detection error rates (YOLO) - FOVEAL - GOOGLE
[google_foveal_detection_error_rate] = detection_error_rates(google_foveal_sigmas,google_foveal_threshs,images_number,google_foveal_detections,gt_detections,detections_resolution,top_k,overlap_correct);

% get detection error rates (YOLO) - FOVEAL - VGG
[vgg_foveal_detection_error_rate] = detection_error_rates(vgg_foveal_sigmas,vgg_foveal_threshs,images_number,vgg_foveal_detections,gt_detections,detections_resolution,top_k,overlap_correct);



% Second passage
% get detection error rates (YOLT) - FOVEAL - CAFFE
[foveal2_detection_error_rate] = detection_error_rates(feedback_sigmas,feedback_threshs,images_number,feedback_detections,gt_detections,detections_resolution,top_k,overlap_correct);

% get detection error rates (YOLT) - FOVEAL - GOOGLE
[google_foveal2_detection_error_rate] = detection_error_rates(google_feedback_sigmas,google_feedback_threshs,images_number,google_feedback_detections,gt_detections,detections_resolution,top_k,overlap_correct);

% get detection error rates (YOLT) - FOVEAL - VGG
[vgg_foveal2_detection_error_rate] = detection_error_rates(vgg_feedback_sigmas,vgg_feedback_threshs,images_number,vgg_feedback_detections,gt_detections,detections_resolution,top_k,overlap_correct);




%% CLASSIFICATION

% First passage
% get classification error rates (YOLO) - FOVEAL - CAFFE
[foveal_top1_classification_error_rate, foveal_top5_classification_error_rate] = classification_error_rates(foveal_sigmas,foveal_threshs,images_number,foveal_classes,gt_classes,top_k);

% get classification error rates (YOLO) - FOVEAL - GOOGLE
[google_foveal_top1_classification_error_rate, google_foveal_top5_classification_error_rate] = classification_error_rates(google_foveal_sigmas,google_foveal_threshs,images_number,google_foveal_classes,gt_classes,top_k);

% get classification error rates (YOLO) - FOVEAL - VGG
[vgg_foveal_top1_classification_error_rate, vgg_foveal_top5_classification_error_rate] = classification_error_rates(vgg_foveal_sigmas,vgg_foveal_threshs,images_number,vgg_foveal_classes,gt_classes,top_k);


% Second passage
% get classification error rates (YOLT) - FOVEAL - CAFFE
[foveal2_top1_classification_error_rate, foveal2_top5_classification_error_rate] = classification_error_rates(feedback_sigmas,feedback_threshs,images_number,feedback_classes,gt_classes,top_k);

% get classification error rates (YOLT) - FOVEAL - GOOGLE
[google_foveal2_top1_classification_error_rate, google_foveal2_top5_classification_error_rate] = classification_error_rates(google_feedback_sigmas,google_feedback_threshs,images_number,google_feedback_classes,gt_classes,top_k);

% get classification error rates (YOLT) - FOVEAL - VGG
[vgg_foveal2_top1_classification_error_rate, vgg_foveal2_top5_classification_error_rate] = classification_error_rates(vgg_feedback_sigmas,vgg_feedback_threshs,images_number,vgg_feedback_classes,gt_classes,top_k);



%% LOCALIZATION ERROR PLOTS - FOVEAL - 1 VERSUS 2 PASSAGE


localization_legend = {...
    char('CaffeNet (1 pass)');...
    char('CaffeNet (2 pass)');...
    char('VGGNet (1 pass)');...
    char('VGGNet (2 pass)');...
    char('GoogLeNet (1 pass)');...
    char('GoogLeNet (2 pass)');...
    };

figure(1)
fontsize=30;
set(gcf, 'Color', [1,1,1]);  % 
plot(foveal_threshs,100*foveal_detection_error_rate(11,:), 'r-o');  
hold on
plot(foveal_threshs,100*foveal2_detection_error_rate(11,:), 'r--o'); 
plot(foveal_threshs,100*vgg_foveal_detection_error_rate(11,:), 'g-*');
plot(foveal_threshs,100*vgg_foveal2_detection_error_rate(11,:), 'g--*');   
plot(foveal_threshs,100*google_foveal_detection_error_rate(11,:), 'b-s');
plot(foveal_threshs,100*google_foveal2_detection_error_rate(11,:), 'b--s'); 
 
xlabel('$th$','Interpreter','LaTex','FontSize',fontsize);
ylabel('Localization Error (%)','Interpreter','LaTex','FontSize',fontsize);
xlim([0 0.95])
ylim([0 100])
set(gca, 'YTick', [0:20:100], 'FontSize', fontsize);
legend('show', 'DislpayName', localization_legend(:) ,'Location', 'southwest');
saveas(figure(1), 'localization_error_foveal_1vs2.png');
%export_fig localization_error_foveal_1vs2 -pdf 



%% CLASSIFICATION ERROR PLOTS - FOVEAL - 1 VERSUS 2 PASSAGE - TOP 1

classification_legend = {...
    char('CaffeNet (1 pass)');...
    char('CaffeNet (2 pass)');...
    char('VGGNet (1 pass)');...
    char('VGGNet (2 pass)');...
    char('GoogLeNet (1 pass)');...
    char('GoogLeNet (2 pass)');...
    };


figure(2)
fontsize=30;
set(gcf, 'Color', [1,1,1]);
plot(foveal_sigmas,100*foveal_top1_classification_error_rate(:,1),'r-o');
hold on
plot(foveal_sigmas,100*foveal2_top1_classification_error_rate(:,1),'r--o');
plot(foveal_sigmas,100*vgg_foveal_top1_classification_error_rate(:,1),'g-*');
plot(foveal_sigmas,100*vgg_foveal2_top1_classification_error_rate(:,1),'g--*'); 
plot(foveal_sigmas,100*google_foveal_top1_classification_error_rate(:,1),'b-s');
plot(foveal_sigmas,100*google_foveal2_top1_classification_error_rate(:,1),'b--s');

xlabel('$\sigma_f$','Interpreter','LaTex','FontSize',fontsize);
ylabel('Classification Error (%)','Interpreter','LaTex','FontSize',fontsize);
ylim([0 100])
set(gca, 'XTick',[0:10:100], 'YTick',[0:20:100], 'FontSize', fontsize);
legend(classification_legend(:),'Location', 'southwest');  % southeast
saveas(figure(2),'classification_error_foveal_1vs2_top1.png')
%export_fig classification_error_foveal_1vs2_top1 -pdf



%% CLASSIFICATION ERROR PLOTS - FOVEAL - 1 VERSUS 2 PASSAGE - TOP 5

classification_legend = {...
    char('CaffeNet (1 pass)');...
    char('CaffeNet (2 pass)');...
    char('VGGNet (1 pass)');...
    char('VGGNet (2 pass)');...
    char('GoogLeNet (1 pass)');...
    char('GoogLeNet (2 pass)');...
    };


figure(3)
fontsize=30;
set(gcf, 'Color', [1,1,1]);
plot(foveal_sigmas,100*foveal_top5_classification_error_rate(:,1),'r-o');
hold on
plot(foveal_sigmas,100*foveal2_top5_classification_error_rate(:,1),'r--o'); 
plot(foveal_sigmas,100*vgg_foveal_top5_classification_error_rate(:,1),'g-*');
plot(foveal_sigmas,100*vgg_foveal2_top5_classification_error_rate(:,1),'g--*'); 
plot(foveal_sigmas,100*google_foveal_top5_classification_error_rate(:,1),'b-s');
plot(foveal_sigmas,100*google_foveal2_top5_classification_error_rate(:,1),'b--s');

xlabel('$\sigma_f$','Interpreter','LaTex','FontSize',fontsize);
ylabel('Classification Error (%)','Interpreter','LaTex','FontSize',fontsize);
ylim([0 100])
set(gca, 'XTick',[0:10:100], 'YTick',[0:20:100], 'FontSize', fontsize);
legend(classification_legend(:),'Location', 'southwest');  % southeast
saveas(figure(3),'classification_error_foveal_1vs2_top5.png')
%export_fig classification_error_foveal_1vs2_top5 -pdf



