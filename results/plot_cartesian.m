close all
addpath('export_fig');

gt_folder='../dataset/gt/';

classifications_file='../files/ground_truth_labels_ilsvrc12.txt';
images_folder='../dataset/images/';

% set to true to check detections
view_detections=false;

% parameters
detections_resolution=227;
images_number=50; 
overlap_correct=0.5;
top_k=5;


%% Load data

% Caffe
cartesian_detection ='../dataset/detections/new/raw_bbox_parse_cartesian_caffe.txt';
cartesian2_detection ='../dataset/detections/new/feedback_detection_parse_cartesian_caffe.txt';

% Google
google_cartesian_detection ='../dataset/detections/new/raw_bbox_parse_cartesian_google.txt';
google_cartesian2_detection ='../dataset/detections/new/feedback_detection_parse_cartesian_google.txt';

% VGG
vgg_cartesian_detection ='../dataset/detections/new/raw_bbox_parse_cartesian_vgg.txt';
vgg_cartesian2_detection ='../dataset/detections/new/feedback_detection_parse_cartesian_vgg.txt';


%% DETECTIONS - First Passage

% get ground truth
[gt_detections, gt_classes]=parse_ground_truth(gt_folder,classifications_file,images_number);

% get detections (YOLO) - CARTESIAN - CAFFE
[cartesian_sigmas,cartesian_threshs,cartesian_classes,cartesian_scores,cartesian_detections]=parse_detections(...
    images_number,...
    cartesian_detection);

% get detections (YOLO) - CARTESIAN - GOOGLE
[google_cartesian_sigmas,google_cartesian_threshs,google_cartesian_classes,google_cartesian_scores,google_cartesian_detections]=parse_detections(...
    images_number,...
    google_cartesian_detection);
  
% get detections (YOLO) - CARTESIAN - VGG
[vgg_cartesian_sigmas,vgg_cartesian_threshs,vgg_cartesian_classes,vgg_cartesian_scores,vgg_cartesian_detections]=parse_detections(...
    images_number,...
    vgg_cartesian_detection);


%% DETECTIONS - Second Passage

% get detections (YOLT) - CARTESIAN - CAFFE
[cartesian_feedback_sigmas,cartesian_feedback_threshs,cartesian_feedback_classes,cartesian_feedback_scores,cartesian_feedback_detections]=parse_detections(...
    images_number,...
    cartesian2_detection);

% get detections (YOLT) - CARTESIAN - GOOGLE
[google_cartesian_feedback_sigmas,google_cartesian_feedback_threshs,google_cartesian_feedback_classes,google_cartesian_feedback_scores,google_cartesian_feedback_detections]=parse_detections(...
    images_number,...
    google_cartesian2_detection);
 
% get detections (YOLT) - CARTESIAN - VGG
[vgg_cartesian_feedback_sigmas,vgg_cartesian_feedback_threshs,vgg_cartesian_feedback_classes,vgg_cartesian_feedback_scores,vgg_cartesian_feedback_detections]=parse_detections(...
    images_number,...
    vgg_cartesian2_detection);



%% LOCALIZATION - First Passage

% get detection error rates (YOLO) - CARTESIAN - CAFFE
[cartesian_detection_error_rate] = detection_error_rates(cartesian_sigmas,cartesian_threshs,images_number,cartesian_detections,gt_detections,detections_resolution,top_k,overlap_correct);

% get detection error rates (YOLO) - CARTESIAN - GOOGLE
[google_cartesian_detection_error_rate] = detection_error_rates(google_cartesian_sigmas,google_cartesian_threshs,images_number,google_cartesian_detections,gt_detections,detections_resolution,top_k,overlap_correct);

% get detection error rates (YOLO) - CARTESIAN - VGG
[vgg_cartesian_detection_error_rate] = detection_error_rates(vgg_cartesian_sigmas,vgg_cartesian_threshs,images_number,vgg_cartesian_detections,gt_detections,detections_resolution,top_k,overlap_correct);



%% LOCALIZATION - Second Passage

% get detection error rates (YOLT) - CARTESIAN - CAFFE
[cartesian_feedback_detection_error_rate] = detection_error_rates(cartesian_feedback_sigmas,cartesian_feedback_threshs,images_number,cartesian_feedback_detections,gt_detections,detections_resolution,top_k,overlap_correct);

% get detection error rates (YOLT) - CARTESIAN - GOOGLE
[google_cartesian_feedback_detection_error_rate] = detection_error_rates(google_cartesian_feedback_sigmas,google_cartesian_feedback_threshs,images_number,google_cartesian_feedback_detections,gt_detections,detections_resolution,top_k,overlap_correct);

% get detection error rates (YOLT) - CARTESIAN - VGG
[vgg_cartesian_feedback_detection_error_rate] = detection_error_rates(vgg_cartesian_feedback_sigmas,vgg_cartesian_feedback_threshs,images_number,vgg_cartesian_feedback_detections,gt_detections,detections_resolution,top_k,overlap_correct);




%% LOCALIZATION ERROR PLOTS - CARTESIAN - 1 VERSUS 2 PASSAGE

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
plot(cartesian_feedback_threshs,100*cartesian_detection_error_rate(5,:), 'r-o');  
hold on
plot(cartesian_feedback_threshs,100*cartesian_feedback_detection_error_rate(5,:), 'r--o'); 
plot(vgg_cartesian_feedback_threshs,100*vgg_cartesian_detection_error_rate(5,:), 'g-*');  
plot(vgg_cartesian_feedback_threshs,100*vgg_cartesian_feedback_detection_error_rate(5,:), 'g--*');   
plot(google_cartesian_feedback_threshs,100*google_cartesian_detection_error_rate(5,:), 'b-s');
plot(google_cartesian_feedback_threshs,100*google_cartesian_feedback_detection_error_rate(5,:), 'b--s'); 
 
xlabel('$th$','Interpreter','LaTex','FontSize',fontsize);
ylabel('Localization Error (%)','Interpreter','LaTex','FontSize',fontsize);
xlim([0 0.95])
ylim([0 100])
set(gca, 'YTick', [0:20:100], 'FontSize', fontsize);
legend('show', 'DislpayName', localization_legend(:) ,'Location', 'southwest');
saveas(figure(1), 'localization_error_cartesian_1vs2.png');
%export_fig localization_error_cartesian_1vs2 -pdf 





%% CLASSIFICATION - First Passage

% get classification error rates (YOLO) -  CARTESIAN - CAFFE
[cartesian_top1_classification_error_rate, cartesian_top5_classification_error_rate] = classification_error_rates(cartesian_sigmas,cartesian_threshs,images_number,cartesian_classes,gt_classes,top_k);

% get classification error rates (YOLO) -  CARTESIAN - GOOGLE
[google_cartesian_top1_classification_error_rate, google_cartesian_top5_classification_error_rate] = classification_error_rates(google_cartesian_sigmas,google_cartesian_threshs,images_number,google_cartesian_classes,gt_classes,top_k);

% get classification error rates (YOLO) -  CARTESIAN - VGG
[vgg_cartesian_top1_classification_error_rate, vgg_cartesian_top5_classification_error_rate] = classification_error_rates(vgg_cartesian_sigmas,vgg_cartesian_threshs,images_number,vgg_cartesian_classes,gt_classes,top_k);



%% CLASSIFICATION - Second Passage

% get classification error rates (YOLT) -  CARTESIAN - CAFFE
[cartesian_feedback_top1_classification_error_rate, cartesian_feedback_top5_classification_error_rate] = classification_error_rates(cartesian_feedback_sigmas,cartesian_feedback_threshs,images_number,cartesian_feedback_classes,gt_classes,top_k);

% get classification error rates (YOLT) -  CARTESIAN - GOOGLE
[google_cartesian_feedback_top1_classification_error_rate, google_cartesian_feedback_top5_classification_error_rate] = classification_error_rates(google_cartesian_feedback_sigmas,google_cartesian_feedback_threshs,images_number,google_cartesian_feedback_classes,gt_classes,top_k);

% get classification error rates (YOLT) -  CARTESIAN - VGG
[vgg_cartesian_feedback_top1_classification_error_rate, vgg_cartesian_feedback_top5_classification_error_rate] = classification_error_rates(vgg_cartesian_feedback_sigmas,vgg_cartesian_feedback_threshs,images_number,vgg_cartesian_feedback_classes,gt_classes,top_k);



%% CLASSIFICATION ERROR PLOTS - CARTESIAN - 1 VERSUS 2 PASSAGE - TOP 1

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
plot(cartesian_sigmas,100*cartesian_top1_classification_error_rate(:,1),'r-o');
hold on
plot(cartesian_sigmas,100*cartesian_feedback_top1_classification_error_rate(:,1),'r--o');
plot(vgg_cartesian_sigmas,100*vgg_cartesian_top1_classification_error_rate(:,1),'g-*');
plot(vgg_cartesian_sigmas,100*vgg_cartesian_feedback_top1_classification_error_rate(:,1),'g--*'); 
plot(google_cartesian_sigmas,100*google_cartesian_top1_classification_error_rate(:,1),'b-s');
plot(google_cartesian_sigmas,100*google_cartesian_feedback_top1_classification_error_rate(:,1),'b--s');

xlabel('$\sigma_f$','Interpreter','LaTex','FontSize',fontsize);
ylabel('Classification Error (%)','Interpreter','LaTex','FontSize',fontsize);
ylim([0 100])
set(gca, 'XTick',[0:10:100], 'YTick',[0:20:100], 'FontSize', fontsize);
legend(classification_legend(:),'Location', 'southwest');  % southeast
saveas(figure(2),'classification_error_cartesian_1vs2_top1.png')
%export_fig classification_error_cartesian_1vs2_top1 -pdf



%% CLASSIFICATION ERROR PLOTS - CARTESIAN - 1 VERSUS 2 PASSAGE - TOP 5

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
plot(cartesian_sigmas,100*cartesian_top5_classification_error_rate(:,1),'r-o');
hold on
plot(cartesian_sigmas,100*cartesian_feedback_top5_classification_error_rate(:,1),'r--o'); 
plot(vgg_cartesian_sigmas,100*vgg_cartesian_top5_classification_error_rate(:,1),'g-*');
plot(vgg_cartesian_sigmas,100*vgg_cartesian_feedback_top5_classification_error_rate(:,1),'g--*'); 
plot(google_cartesian_sigmas,100*google_cartesian_top5_classification_error_rate(:,1),'b-s');
plot(google_cartesian_sigmas,100*google_cartesian_feedback_top5_classification_error_rate(:,1),'b--s');

xlabel('$\sigma_f$','Interpreter','LaTex','FontSize',fontsize);
ylabel('Classification Error (%)','Interpreter','LaTex','FontSize',fontsize);
ylim([0 100])
set(gca, 'XTick',[0:10:100], 'YTick',[0:20:100], 'FontSize', fontsize);
legend(classification_legend(:),'Location', 'southwest');  % southeast
saveas(figure(3),'classification_error_cartesian_1vs2_top5.png')
%export_fig classification_error_cartesian_1vs2_top5 -pdf
