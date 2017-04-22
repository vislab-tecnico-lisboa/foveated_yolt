close all
addpath('export_fig');

gt_folder='../dataset/gt/';

%% Foveal
% First passage 
foveal_detection ='../dataset/detections/new/raw_bbox_parse_foveal_vgg.txt';

% Second passage 
foveal2_detection ='../dataset/detections/new/feedback_detection_foveal_vgg.txt';


%% CARTESIAN
% First passage
cartesian_detection ='../dataset/detections/new/raw_bbox_parse_cartesian_vgg.txt';

% Second passage
cartesian2_detection ='../dataset/detections/new/feedback_detection_parse_cartesian_vgg.txt';



%% HYBRID
% First passage
hybrid_detection ='../dataset/detections/new/raw_bbox_parse_vale_high_blur_vggnet_100.txt';

% Second passage
hybrid2_detection ='../dataset/detections/new/feedback_detection_parse_vale_high_blur_vggnet_100.txt';


classifications_file='../files/ground_truth_labels_ilsvrc12.txt';
images_folder='../dataset/images/';

% set to true to check detections
view_detections=false;

% parameters
detections_resolution=227;
images_number=100; 
overlap_correct=0.5;
top_k=5;


%% DETECTIONS - First passage

% get ground truth
[gt_detections, gt_classes]=parse_ground_truth(gt_folder,classifications_file,images_number);

% get detections (YOLO) - FOVEAL - VGG
[foveal_sigmas,foveal_threshs,foveal_classes,foveal_scores,foveal_detections]=parse_detections(...
    images_number,...
    foveal_detection);

% get detections (YOLO) - CARTESIAN - VGG
[cartesian_sigmas,cartesian_threshs,cartesian_classes,cartesian_scores,cartesian_detections]=parse_detections(...
    50,...
    cartesian_detection);

% get detections (YOLO) - HYBRID - VGG
[hybrid_sigmas,hybrid_threshs,hybrid_classes,hybrid_scores,hybrid_detections]=parse_detections(...
     images_number,...
     hybrid_detection);


%% DETECTIONS - Second passage

% get detections (YOLT) - FOVEAL - VGG 
[feedback_sigmas,feedback_threshs,feedback_classes,feedback_scores,rank_feedback_classes,feedback_detections]=feedback_parse_detections2(...
    images_number,...
    foveal2_detection);

% get detections (YOLT) - CARTESIAN - VGG
[cartesian_feedback_sigmas,cartesian_feedback_threshs,cartesian_feedback_classes,cartesian_feedback_scores,cartesian_feedback_detections]=parse_detections(...
    50,...
    cartesian2_detection);

% get detections (YOLT) - HYBRID - VGG 
[hybrid_feedback_sigmas,hybrid_feedback_threshs,hybrid_feedback_classes,hybrid_feedback_scores,hybrid_rank_feedback_classes,hybrid_feedback_detections]=feedback_parse_detections2(...
    images_number,...
    hybrid2_detection);

%% LOCALIZATION

% First passage
% get detection error rates (YOLO) - FOVEAL - VGG
[foveal_detection_error_rate] = detection_error_rates(foveal_sigmas,foveal_threshs,images_number,foveal_detections,gt_detections,detections_resolution,top_k,overlap_correct);

% Second passage
% get detection error rates (YOLT) - FOVEAL - VGG
[foveal2_detection_error_rate] = detection_error_rates(feedback_sigmas,feedback_threshs,images_number,feedback_detections,gt_detections,detections_resolution,top_k,overlap_correct);


% get detection error rates (YOLO) - CARTESIAN - VGG
[cartesian_detection_error_rate] = detection_error_rates(cartesian_sigmas,cartesian_threshs,50,cartesian_detections,gt_detections,detections_resolution,top_k,overlap_correct);

% get detection error rates (YOLT) - CARTESIAN - VGG
[cartesian_feedback_detection_error_rate] = detection_error_rates(cartesian_feedback_sigmas,cartesian_feedback_threshs,50,cartesian_feedback_detections,gt_detections,detections_resolution,top_k,overlap_correct);



% get detection error rates (YOLO) - HYBRID - VGG
[hybrid_detection_error_rate] = detection_error_rates(hybrid_sigmas,hybrid_threshs,images_number,hybrid_detections,gt_detections,detections_resolution,top_k,overlap_correct);

% get detection error rates (YOLT) - HYBRID - VGG
[hybrid2_detection_error_rate] = detection_error_rates(hybrid_feedback_sigmas,hybrid_feedback_threshs,images_number,hybrid_feedback_detections,gt_detections,detections_resolution,top_k,overlap_correct);


%% CLASSIFICATION

% First passage
% get classification error rates (YOLO) - CAFFE - FOVEAL - VGG
[foveal_top1_classification_error_rate, foveal_top5_classification_error_rate] = classification_error_rates(foveal_sigmas,foveal_threshs,images_number,foveal_classes,gt_classes,top_k);


% Second passage
% get classification error rates (YOLT) - CAFFE - FOVEAL - VGG
[foveal2_top1_classification_error_rate, foveal2_top5_classification_error_rate] = classification_error_rates(feedback_sigmas,feedback_threshs,images_number,feedback_classes,gt_classes,top_k);



% get classification error rates (YOLO) -  CARTESIAN - VGG
[cartesian_top1_classification_error_rate, cartesian_top5_classification_error_rate] = classification_error_rates(cartesian_sigmas,cartesian_threshs,50,cartesian_classes,gt_classes,top_k);

% get classification error rates (YOLT) -  CARTESIAN - VGG
[cartesian_feedback_top1_classification_error_rate, cartesian_feedback_top5_classification_error_rate] = classification_error_rates(cartesian_feedback_sigmas,cartesian_feedback_threshs,50,cartesian_feedback_classes,gt_classes,top_k);


% get classification error rates (YOLO) - CAFFE - HYBRID - VGG
[hybrid_top1_classification_error_rate, hybrid_top5_classification_error_rate] = classification_error_rates(hybrid_sigmas,hybrid_threshs,images_number,hybrid_classes,gt_classes,top_k);

% get classification error rates (YOLT) - CAFFE - HYBRID - VGG
[hybrid2_top1_classification_error_rate, hybrid2_top5_classification_error_rate] = classification_error_rates(hybrid_feedback_sigmas,hybrid_feedback_threshs,images_number,hybrid_feedback_classes,gt_classes,top_k);




%% PLOT

%% LOCALIZATION
% FIRST PASS
localization_legend = {...
    char('Foveal (1 pass)');...
    char('Cartesian (1 pass)');...
    char('Combined (1 pass)');...
    };

figure(1)
fontsize=30;
set(gcf, 'Color', [1,1,1]);  % 
plot(foveal_threshs,100*foveal_detection_error_rate(11,:), 'r-o');   
hold on
plot(cartesian_threshs,100*cartesian_detection_error_rate(5,:), 'g-*');   
plot(hybrid_threshs,100*hybrid_detection_error_rate(9,:), 'b-s');   

xlabel('$th$','Interpreter','LaTex','FontSize',fontsize);
ylabel('Localization Error (%)','Interpreter','LaTex','FontSize',fontsize);
xlim([0 0.95])
ylim([0 100])
set(gca, 'FontSize', fontsize);
set(gca,'YTick',[0:20:100], 'FontSize', fontsize);
legend('show', 'DislpayName', localization_legend(:) ,'Location', 'southwest');
saveas(figure(1), 'localization_error_vgg_all_models.png');
%export_fig localization_error_vgg_all_models -pdf 


% SECOND PASS
localization_legend = {...
    char('Foveal (2 pass)');...
    char('Cartesian (2 pass)');...
    char('Combined (2 pass)');...
    };

figure(2)
fontsize=30;
set(gcf, 'Color', [1,1,1]);  % 
plot(foveal_threshs,100*foveal2_detection_error_rate(11,:), 'r-o');   
hold on
plot(cartesian_threshs,100*cartesian_feedback_detection_error_rate(5,:), 'g-*');   
plot(hybrid_threshs,100*hybrid2_detection_error_rate(9,:), 'b-s');   

xlabel('$th$','Interpreter','LaTex','FontSize',fontsize);
ylabel('Localization Error (%)','Interpreter','LaTex','FontSize',fontsize);
xlim([0 0.95])
ylim([0 100])
set(gca, 'FontSize', fontsize);
set(gca,'YTick',[0:20:100], 'FontSize', fontsize);
legend('show', 'DislpayName', localization_legend(:) ,'Location', 'southwest');
saveas(figure(2), 'localization_error_vgg_feedback_all_models.png');
%export_fig localization_error_vgg_feedback_all_models -pdf 







%% CLASSIFICATION
% FIRST PASS
classification_legend = {...
    char('Foveal ');...
    %char('Foveal ');...
    char('Cartesian ');...
    %char('Cartesian ');...
    char('Combined ');...
    %char('Combined ');...
    };


figure(3)
fontsize=30;
set(gcf, 'Color', [1,1,1]);
plot(foveal_sigmas,100*foveal_top1_classification_error_rate(:,1),'r-o');
hold on
plot(cartesian_sigmas,100*cartesian_top1_classification_error_rate(:,1),'g-s');
plot(hybrid_sigmas,100*hybrid_top1_classification_error_rate(:,1),'b-*');
plot(foveal_sigmas,100*foveal_top5_classification_error_rate(:,1),'r--o');
plot(cartesian_sigmas,100*cartesian_top5_classification_error_rate(:,1),'g--s'); 
plot(hybrid_sigmas,100*hybrid_top5_classification_error_rate(:,1),'b--*');

xlabel('$\sigma_f$','Interpreter','LaTex','FontSize',fontsize);
ylabel('Classification Error (%)','Interpreter','LaTex','FontSize',fontsize);
ylim([0 100])
set(gca, 'XTick',[0:10:100], 'YTick',[0:20:100], 'FontSize', fontsize);
legend(classification_legend(:),'Location', 'southwest');  % southeast
saveas(figure(3),'classification_error_vgg_all_models.png')
%export_fig classification_error_vgg_all_models -pdf


% SECOND PASS
classification_legend = {...
    char('Foveal (2 pass)');...
    %char('Foveal top-5 (2 pass)');...
    char('Cartesian (2 pass)');...
    %char('Cartesian top-5 (2 pass)');...
    char('Combined (2 pass)');...
    %char('Combined top-5 (2 pass)');...
    };


figure(4)
fontsize=30; 
set(gcf, 'Color', [1,1,1]);
plot(foveal_sigmas,100*foveal2_top1_classification_error_rate(:,1),'r-o');
hold on
plot(cartesian_sigmas,100*cartesian_feedback_top1_classification_error_rate(:,1),'g-s');
plot(hybrid_sigmas,100*hybrid2_top1_classification_error_rate(:,1),'b-*'); 
plot(foveal_sigmas,100*foveal2_top5_classification_error_rate(:,1),'r--o');
plot(cartesian_sigmas,100*cartesian_feedback_top5_classification_error_rate(:,1),'g--s'); 
plot(hybrid_sigmas,100*hybrid2_top5_classification_error_rate(:,1),'b--*');

xlabel('$\sigma_f$','Interpreter','LaTex','FontSize',fontsize);
ylabel('Classification Error (%)','Interpreter','LaTex','FontSize',fontsize);
ylim([0 100])
set(gca, 'XTick',[0:10:100], 'YTick',[0:20:100], 'FontSize', fontsize);
legend(classification_legend(:),'Location', 'southwest');  % southeast
saveas(figure(4),'classification_error_vgg_feedback_all_models.png')
%export_fig classification_error_vgg_feedback_all_models -pdf


