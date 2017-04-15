close all
addpath('export_fig');

gt_folder='../dataset/gt/';

%% First passage
foveal_detection ='../dataset/detections/new/raw_bbox_parse_foveal_caffe2.txt';
google_foveal_detection ='../dataset/detections/new/raw_bbox_parse_foveal_google2.txt';
vgg_foveal_detection ='../dataset/detections/new/raw_bbox_parse_foveal_vgg.txt';


%% Second passage
foveal2_detection ='../dataset/detections/new/feedback_detection_foveal_caffe2.txt';
google_foveal2_detection ='../dataset/detections/new/feedback_detection_foveal_google2.txt';
vgg_foveal2_detection ='../dataset/detections/new/feedback_detection_foveal_vgg.txt';


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



%% VIEW DETECTIONS (BBOX)

% view images
if view_detections
    for i=1:images_number
        figure(i)
        imshow(strcat(images_folder,gt_detections(i).filename))
        hold on
        for g=1:size(gt_detections(i).bboxes,1)
            % gt bbox
            gt_bbox=gt_detections(i).bboxes(g,:);
            rectangle('Position',...
                gt_bbox,...
                'EdgeColor',...
                [0 1 0],...
                'LineWidth',...
                3);
        end
        hold off
    end
end



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
% get classification error rates (YOLO) - CAFFE - FOVEAL - CAFFE
[foveal_top1_classification_error_rate, foveal_top5_classification_error_rate] = classification_error_rates(foveal_sigmas,foveal_threshs,images_number,foveal_classes,gt_classes,top_k);

% get classification error rates (YOLO) - CAFFE - FOVEAL - GOOGLE
[google_foveal_top1_classification_error_rate, google_foveal_top5_classification_error_rate] = classification_error_rates(google_foveal_sigmas,google_foveal_threshs,images_number,google_foveal_classes,gt_classes,top_k);

% get classification error rates (YOLO) - CAFFE - FOVEAL - VGG
[vgg_foveal_top1_classification_error_rate, vgg_foveal_top5_classification_error_rate] = classification_error_rates(vgg_foveal_sigmas,vgg_foveal_threshs,images_number,vgg_foveal_classes,gt_classes,top_k);


% Second passage
% get classification error rates (YOLT) - CAFFE - FOVEAL - CAFFE
[foveal2_top1_classification_error_rate, foveal2_top5_classification_error_rate] = classification_error_rates(feedback_sigmas,feedback_threshs,images_number,feedback_classes,gt_classes,top_k);

% get classification error rates (YOLT) - CAFFE - FOVEAL - GOOGLE
[google_foveal2_top1_classification_error_rate, google_foveal2_top5_classification_error_rate] = classification_error_rates(google_feedback_sigmas,google_feedback_threshs,images_number,google_feedback_classes,gt_classes,top_k);

% get classification error rates (YOLT) - CAFFE - FOVEAL - VGG
[vgg_foveal2_top1_classification_error_rate, vgg_foveal2_top5_classification_error_rate] = classification_error_rates(vgg_feedback_sigmas,vgg_feedback_threshs,images_number,vgg_feedback_classes,gt_classes,top_k);



%% LOCALIZATION ERROR PLOTS - FOVEAL - Different models - FIRST PASS

% fix one sigma and plot all saliency thresholds
sigma_index=1;

% sigmas_leg = [1 20 40 60 80 100];
% plot_detection = [foveal_detection_error_rate(1,:); foveal_detection_error_rate(3,:); foveal_detection_error_rate(5,:); foveal_detection_error_rate(7,:); foveal_detection_error_rate(9,:); foveal_detection_error_rate(11,:)];
% legend_sigma = {};
% for i=1:length(sigmas_leg)
%     legend_sigma = [legend_sigma, strcat('\sigma=', num2str(sigmas_leg(i))) ];
% end

% localization_legend = {...
%     char('Backward (Foveal) CaffeNet \sigma_f = 80');...
%     char('Backward (Foveal) CaffeNet \sigma_f = 100');...
%     char('Backward (Foveal) VGGNet \sigma_f = 80');...
%     char('Backward (Foveal) VGGNet \sigma_f = 100');...
%     char('Backward (Foveal) GoogLeNet \sigma_f = 80');...
%     char('Backward (Foveal) GoogLeNet \sigma_f = 100');...
%     };

localization_legend = {...
    char('Backward (CaffeNet)');...
    char('Backward (VGGNet)');...
    char('Backward (GoogLeNet)');...
    };

figure(1)
fontsize=30;
set(gcf, 'Color', [1,1,1]);  % 
plot(foveal_threshs,100*foveal_detection_error_rate(11,:), 'r-o');

hold on
plot(foveal_threshs,100*vgg_foveal_detection_error_rate(11,:), 'g-*'); 
plot(foveal_threshs,100*google_foveal_detection_error_rate(11,:), 'b-s'); 
plot(foveal_threshs,100*foveal_detection_error_rate(9,:), 'r--o');   

plot(foveal_threshs,100*vgg_foveal_detection_error_rate(9,:), 'g--*');   
plot(foveal_threshs,100*google_foveal_detection_error_rate(9,:), 'b--s');   


xlabel('$th$','Interpreter','LaTex','FontSize',fontsize);
ylabel('Localization Error (%)','Interpreter','LaTex','FontSize',fontsize);
xlim([0 0.95])
ylim([0 100])
set(gca, 'YTick', [0:20:100], 'FontSize', fontsize);
legend('show', 'DislpayName', localization_legend(:) ,'Location', 'southwest');
%saveas(figure(1), 'localization_error_foveal.png');
export_fig localization_error_foveal -pdf 



%% CLASSIFICATION ERROR PLOTS - FOVEAL - Different models - FIRST PASS

classification_legend = {...

    char('Feed-foward (CaffeNet) ');...
    char('Feed-foward (VGGNet)');...
    char('Feed-foward (GoogLeNet)');...
    };

% classification_legend = {...
%     char('top-1 feed-foward (Foveal) CaffeNet ');...
%     char('top-5 feed-foward (Foveal) CaffeNet ');...
%     char('top-1 feed-foward (Foveal) VGGNet');...
%     char('top-5 feed-foward (Foveal) VGGNet');...
%     char('top-1 feed-foward (Foveal) GoogLeNet');...
%     char('top-5 feed-foward (Foveal) GoogLeNet');...
%     };


figure(2)
%fontsize=30;
set(gcf, 'Color', [1,1,1]);
plot(foveal_sigmas,100*foveal_top5_classification_error_rate(:,1),'r-o');
hold on
plot(foveal_sigmas,100*vgg_foveal_top5_classification_error_rate(:,1),'g-*');
plot(foveal_sigmas,100*google_foveal_top5_classification_error_rate(:,1),'b-s');
plot(foveal_sigmas,100*foveal_top1_classification_error_rate(:,1),'r--o'); 
plot(foveal_sigmas,100*vgg_foveal_top1_classification_error_rate(:,1),'g--*'); 
plot(foveal_sigmas,100*google_foveal_top1_classification_error_rate(:,1),'b--s');

xlabel('$\sigma_f$','Interpreter','LaTex','FontSize',fontsize);
ylabel('Classification Error (%)','Interpreter','LaTex','FontSize',fontsize);
ylim([0 100])
xlim([1 100])
set(gca, 'XTick',[1 10:10:100], 'YTick',[0:20:100], 'FontSize', fontsize);
legend(classification_legend(:),'Location', 'southwest');  % southeast
%saveas(figure(2),'classification_error_foveal.png')
export_fig classification_error_foveal -pdf


%% LOCALIZATION ERROR PLOTS - FOVEAL - Different models - SECOND PASS

% fix one sigma and plot all saliency thresholds
sigma_index=1;

% sigmas_leg = [1 20 40 60 80 100];
% plot_detection = [foveal_detection_error_rate(1,:); foveal_detection_error_rate(3,:); foveal_detection_error_rate(5,:); foveal_detection_error_rate(7,:); foveal_detection_error_rate(9,:); foveal_detection_error_rate(11,:)];
% legend_sigma = {};
% for i=1:length(sigmas_leg)
%     legend_sigma = [legend_sigma, strcat('\sigma=', num2str(sigmas_leg(i))) ];
% end
% 
% feedback_localization_legend = {...
%     char('2º Backward (Foveal) CaffeNet \sigma_f = 80');...
%     char('2º Backward (Foveal) CaffeNet \sigma_f = 100');...
%     char('2º Backward (Foveal) VGGNet \sigma_f = 80');...
%     char('2º Backward (Foveal) VGGNet \sigma_f = 100');...
%     char('2º Backward (Foveal) GoogLeNet \sigma_f = 80');...
%     char('2º Backward (Foveal) GoogLeNet \sigma_f = 100');...
%     };



feedback_localization_legend = {...
    char('Backward (CaffeNet)');...
    char('Backward (VGGNet)');...
    char('Backward (GoogLeNet)');...
    };

figure(3)
%fontsize=30;
set(gcf, 'Color', [1,1,1]);  % 

plot(feedback_threshs,100*foveal2_detection_error_rate(11,:), 'r-o');  
hold on
plot(feedback_threshs,100*vgg_foveal2_detection_error_rate(11,:), 'g-*');
plot(feedback_threshs,100*google_foveal2_detection_error_rate(11,:), 'b-s'); 

plot(feedback_threshs,100*foveal2_detection_error_rate(9,:), 'r--o');   
plot(feedback_threshs,100*vgg_foveal2_detection_error_rate(9,:), 'g--*');   
plot(feedback_threshs,100*google_foveal2_detection_error_rate(9,:), 'b--s');   

 
xlabel('$th$','Interpreter','LaTex','FontSize',fontsize);
ylabel('Localization Error (%)','Interpreter','LaTex','FontSize',fontsize);
xlim([0 0.95])
ylim([0 100])
set(gca, 'YTick',[0:20:100], 'FontSize', fontsize);
legend('show', 'DislpayName', feedback_localization_legend(:) ,'Location', 'southwest');
%saveas(figure(3), 'localization_error_foveal_feedback.png');
export_fig localization_error_foveal_feedback -pdf 



%% CLASSIFICATION ERROR PLOTS - FOVEAL - Different models - SECOND PASS

feedback_classification_legend = {...

    char('Feed-foward (CaffeNet) ');...
    char('Feed-foward (VGGNet) ');...
    char('Feed-foward (GoogLeNet) ');...
    };


% feedback_classification_legend = {...
%     char('top-1 2º feed-foward (Foveal) CaffeNet ');...
%     char('top-5 2º feed-foward (Foveal) CaffeNet ');...
%     char('top-1 2º feed-foward (Foveal) VGGNet');...
%     char('top-5 2º feed-foward (Foveal) VGGNet');...
%     char('top-1 2º feed-foward (Foveal) GoogLeNet');...
%     char('top-5 2º feed-foward (Foveal) GoogLeNet');...
%     };


figure(4)
%fontsize=30;
set(gcf, 'Color', [1,1,1]);
plot(feedback_sigmas,100*foveal2_top5_classification_error_rate(:,1),'r-o');
hold on
plot(feedback_sigmas,100*vgg_foveal2_top5_classification_error_rate(:,1),'g-*');
plot(feedback_sigmas,100*google_foveal2_top5_classification_error_rate(:,1),'b-s');
plot(feedback_sigmas,100*foveal2_top1_classification_error_rate(:,1),'r--o'); 
plot(feedback_sigmas,100*vgg_foveal2_top1_classification_error_rate(:,1),'g--*'); 
plot(feedback_sigmas,100*google_foveal2_top1_classification_error_rate(:,1),'b--s');

plot(feedback_sigmas,100*foveal2_top1_classification_error_rate(:,1),'r--o'); 
plot(feedback_sigmas,100*vgg_foveal2_top1_classification_error_rate(:,1),'g--*'); 

plot(feedback_sigmas,100*google_foveal2_top1_classification_error_rate(:,1),'b--s'); 


xlabel('$\sigma_f$','Interpreter','LaTex','FontSize',fontsize);
ylabel('Classification Error (%)','Interpreter','LaTex','FontSize',fontsize);
ylim([0 100])
xlim([1 100])
set(gca, 'XTick',[1 10:10:100], 'YTick',[0:20:100], 'FontSize', fontsize);
legend(feedback_classification_legend(:),'Location', 'southwest');  % southeast
%saveas(figure(4),'classification_error_foveal_feedback.png')
export_fig classification_error_foveal_feedback -pdf





