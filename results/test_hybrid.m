close all
addpath('export_fig');

gt_folder='../dataset/gt/';

%% First passage
hybrid_detection ='../dataset/detections/new/raw_bbox_parse_vale_high_blur_caffe_100.txt';
google_hybrid_detection ='../dataset/detections/new/raw_bbox_parse_vale_high_blur_google_100.txt';
vgg_hybrid_detection ='../dataset/detections/new/raw_bbox_parse_vale_high_blur_vggnet_100.txt';


%% Second passage
hybrid2_detection ='../dataset/detections/new/feedback_detection_parse_vale_high_blur_caffe_100.txt';
google_hybrid2_detection ='../dataset/detections/new/feedback_detection_parse_vale_high_blur_google_100.txt';
vgg_hybrid2_detection ='../dataset/detections/new/feedback_detection_parse_vale_high_blur_vggnet_100.txt';


classifications_file='../files/ground_truth_labels_ilsvrc12.txt';
images_folder='../dataset/images/';

% set to true to check detections
view_detections=false;

% parameters
detections_resolution=227;
images_number=100; 
overlap_correct=0.5;
top_k=5;


%% DETECTIONS - First passage - Imagem com blur uniform de sigma = 5

% get ground truth
[gt_detections, gt_classes]=parse_ground_truth(gt_folder,classifications_file,images_number);

% get detections (YOLO) - HYBRID - CAFFE
[hybrid_sigmas,hybrid_threshs,hybrid_classes,hybrid_scores,hybrid_detections]=parse_detections(...
    images_number,...
    hybrid_detection);

% get detections (YOLO) - HYBRID - GOOGLE
[google_hybrid_sigmas,google_hybrid_threshs,google_hybrid_classes,google_hybrid_scores,google_hybrid_detections]=parse_detections(...
    images_number,...
    google_hybrid_detection);

% get detections (YOLO) - HYBRID - VGG
[vgg_hybrid_sigmas,vgg_hybrid_threshs,vgg_hybrid_classes,vgg_hybrid_scores,vgg_hybrid_detections]=parse_detections(...
    images_number,...
    vgg_hybrid_detection);

 

%% DETECTIONS - Second passage - Imagem com alta resolu��o que � foveada no centro da bbox (varios sigmas)

% get detections (YOLT) - HYBRID - CAFFE 
[feedback_sigmas,feedback_threshs,feedback_classes,feedback_scores,rank_feedback_classes,feedback_detections]=feedback_parse_detections2(...
    images_number,...
    hybrid2_detection);

% get detections (YOLT) - HYBRID - GOOGLE 
[google_feedback_sigmas,google_feedback_threshs,google_feedback_classes,google_feedback_scores,google_rank_feedback_classes,google_feedback_detections]=feedback_parse_detections2(...
    images_number,...
    google_hybrid2_detection);

% get detections (YOLT) - HYBRID - VGG 
[vgg_feedback_sigmas,vgg_feedback_threshs,vgg_feedback_classes,vgg_feedback_scores,vgg_rank_feedback_classes,vgg_feedback_detections]=feedback_parse_detections2(...
    images_number,...
    vgg_hybrid2_detection);



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
% get detection error rates (YOLO) - HYBRID - CAFFE
[hybrid_detection_error_rate] = detection_error_rates(hybrid_sigmas,hybrid_threshs,images_number,hybrid_detections,gt_detections,detections_resolution,top_k,overlap_correct);

% get detection error rates (YOLO) - HYBRID - GOOGLE
[google_hybrid_detection_error_rate] = detection_error_rates(google_hybrid_sigmas,google_hybrid_threshs,images_number,google_hybrid_detections,gt_detections,detections_resolution,top_k,overlap_correct);

% get detection error rates (YOLO) - HYBRID - VGG
[vgg_hybrid_detection_error_rate] = detection_error_rates(vgg_hybrid_sigmas,vgg_hybrid_threshs,images_number,vgg_hybrid_detections,gt_detections,detections_resolution,top_k,overlap_correct);



% Second passage
% get detection error rates (YOLT) - HYBRID - CAFFE
[hybrid2_detection_error_rate] = detection_error_rates(feedback_sigmas,feedback_threshs,images_number,feedback_detections,gt_detections,detections_resolution,top_k,overlap_correct);

% get detection error rates (YOLT) - HYBRID - GOOGLE
[google_hybrid2_detection_error_rate] = detection_error_rates(google_feedback_sigmas,google_feedback_threshs,images_number,google_feedback_detections,gt_detections,detections_resolution,top_k,overlap_correct);

% get detection error rates (YOLT) - HYBRID - VGG
[vgg_hybrid2_detection_error_rate] = detection_error_rates(vgg_feedback_sigmas,vgg_feedback_threshs,images_number,vgg_feedback_detections,gt_detections,detections_resolution,top_k,overlap_correct);



%% CLASSIFICATION

% First passage
% get classification error rates (YOLO) - CAFFE - HYBRID - CAFFE
[hybrid_top1_classification_error_rate, hybrid_top5_classification_error_rate] = classification_error_rates(hybrid_sigmas,hybrid_threshs,images_number,hybrid_classes,gt_classes,top_k);

% get classification error rates (YOLO) - CAFFE - HYBRID - GOOGLE
[google_hybrid_top1_classification_error_rate, google_hybrid_top5_classification_error_rate] = classification_error_rates(google_hybrid_sigmas,google_hybrid_threshs,images_number,google_hybrid_classes,gt_classes,top_k);

% get classification error rates (YOLO) - CAFFE - HYBRID - VGG
[vgg_hybrid_top1_classification_error_rate, vgg_hybrid_top5_classification_error_rate] = classification_error_rates(vgg_hybrid_sigmas,vgg_hybrid_threshs,images_number,vgg_hybrid_classes,gt_classes,top_k);


% Second passage
% get classification error rates (YOLT) - CAFFE - HYBRID - CAFFE
[hybrid2_top1_classification_error_rate, hybrid2_top5_classification_error_rate] = classification_error_rates(feedback_sigmas,feedback_threshs,images_number,feedback_classes,gt_classes,top_k);

% get classification error rates (YOLT) - CAFFE - HYBRID - GOOGLE
[google_hybrid2_top1_classification_error_rate, google_hybrid2_top5_classification_error_rate] = classification_error_rates(google_feedback_sigmas,google_feedback_threshs,images_number,google_feedback_classes,gt_classes,top_k);

% get classification error rates (YOLT) - CAFFE - HYBRID - VGG
[vgg_hybrid2_top1_classification_error_rate, vgg_hybrid2_top5_classification_error_rate] = classification_error_rates(vgg_feedback_sigmas,vgg_feedback_threshs,images_number,vgg_feedback_classes,gt_classes,top_k);



%% LOCALIZATION ERROR PLOTS - HYBRID - Different models - FIRST PASS

% fix one sigma and plot all saliency thresholds
sigma_index=1;

% sigmas_leg = [1 20 40 60 80 100];
% plot_detection = [foveal_detection_error_rate(1,:); foveal_detection_error_rate(3,:); foveal_detection_error_rate(5,:); foveal_detection_error_rate(7,:); foveal_detection_error_rate(9,:); foveal_detection_error_rate(11,:)];
% legend_sigma = {};
% for i=1:length(sigmas_leg)
%     legend_sigma = [legend_sigma, strcat('\sigma=', num2str(sigmas_leg(i))) ];
% end

localization_legend = {...
    'Backward CaffeNet (\sigma_u = 5)';...
    'Backward VGGNet (\sigma_u = 5)';...
    'Backward GoogLeNet (\sigma_u = 5)';...
    };

figure(1)
fontsize=30;
set(gcf, 'Color', [1,1,1]);  % 
plot(hybrid_threshs,100*hybrid_detection_error_rate(9,:), 'r--o');   
hold on
plot(hybrid_threshs,100*vgg_hybrid_detection_error_rate(9,:), 'g--*');   
plot(hybrid_threshs,100*google_hybrid_detection_error_rate(9,:), 'b--s');   

xlabel('$th$','Interpreter','LaTex','FontSize',fontsize);
ylabel('Localization Error (%)','Interpreter','LaTex','FontSize',fontsize);
xlim([0 0.95])
ylim([0 100])
set(gca, 'FontSize', fontsize);
set(gca,'YTick',[0:20:100], 'FontSize', fontsize);
legend('show', 'DislpayName', localization_legend(:) ,'Location', 'southwest');
%saveas(figure(1), 'localization_error_hybrid.png');
export_fig localization_error_hybrid -pdf 



%% CLASSIFICATION ERROR PLOTS - HYBRID - Different models - FIRST PASS

classification_legend = {...
    char('top-1 feed-foward (Foveal) CaffeNet ');...
    char('top-5 feed-foward (Foveal) CaffeNet ');...
    char('top-1 feed-foward (Foveal) VGGNet');...
    char('top-5 feed-foward (Foveal) VGGNet');...
    char('top-1 feed-foward (Foveal) GoogLeNet');...
    char('top-5 feed-foward (Foveal) GoogLeNet');...
    };


figure(2)
set(gcf, 'Color', [1,1,1]);
plot(hybrid_sigmas,100*hybrid_top1_classification_error_rate(:,1),'r--o'); 
hold on
plot(hybrid_sigmas,100*hybrid_top5_classification_error_rate(:,1),'r-o');
plot(hybrid_sigmas,100*vgg_hybrid_top1_classification_error_rate(:,1),'g--*'); 
plot(hybrid_sigmas,100*vgg_hybrid_top5_classification_error_rate(:,1),'g-*');
plot(hybrid_sigmas,100*google_hybrid_top1_classification_error_rate(:,1),'b--s'); 
plot(hybrid_sigmas,100*google_hybrid_top5_classification_error_rate(:,1),'b-s');
ylim([0 100])

xlabel('$\sigma_u$','Interpreter','LaTex','FontSize',fontsize);
ylabel('Classification Error (%)','Interpreter','LaTex','FontSize',fontsize);
set(gca,'XTick',[0:20:100], 'FontSize', fontsize);
set(gca,'YTick',[0:20:100], 'FontSize', fontsize);
%legend(classification_legend(:),'Location', 'southwest');  % southeast
%saveas(figure(2),'classification_error_hybrid.png')
export_fig classification_error_hybrid -pdf








%% LOCALIZATION ERROR PLOTS - HYBRID - Different models - SECOND PASS

% fix one sigma and plot all saliency thresholds
sigma_index=1;

% sigmas_leg = [1 20 40 60 80 100];
% plot_detection = [foveal_detection_error_rate(1,:); foveal_detection_error_rate(3,:); foveal_detection_error_rate(5,:); foveal_detection_error_rate(7,:); foveal_detection_error_rate(9,:); foveal_detection_error_rate(11,:)];
% legend_sigma = {};
% for i=1:length(sigmas_leg)
%     legend_sigma = [legend_sigma, strcat('\sigma=', num2str(sigmas_leg(i))) ];
% end

feedback_localization_legend = {...
    char('2� Backward (Foveal) CaffeNet \sigma = 80');...
    char('2� Backward (Foveal) CaffeNet \sigma = 100');...
    char('2� Backward (Foveal) VGGNet \sigma = 80');...
    char('2� Backward (Foveal) VGGNet \sigma = 100');...
    char('2� Backward (Foveal) GoogLeNet \sigma = 80');...
    char('2� Backward (Foveal) GoogLeNet \sigma = 100');...
    };

figure(3)
fontsize=30;
set(gcf, 'Color', [1,1,1]);  % 
plot(feedback_threshs,100*hybrid2_detection_error_rate(9,:), 'r--o');   
hold on
plot(feedback_threshs,100*hybrid2_detection_error_rate(11,:), 'r-o');  
plot(feedback_threshs,100*vgg_hybrid2_detection_error_rate(9,:), 'g--*');   
plot(feedback_threshs,100*vgg_hybrid2_detection_error_rate(11,:), 'g-*'); 
plot(feedback_threshs,100*google_hybrid2_detection_error_rate(9,:), 'b--s');   
plot(feedback_threshs,100*google_hybrid2_detection_error_rate(11,:), 'b-s'); 

xlabel('$th$','Interpreter','LaTex','FontSize',fontsize);
ylabel('Localization Error (%)','Interpreter','LaTex','FontSize',fontsize);
xlim([0 0.95])
ylim([0 100])
set(gca, 'FontSize', fontsize);
set(gca,'YTick',[0:20:100], 'FontSize', fontsize);
%legend('show', 'DislpayName', feedback_localization_legend(:) ,'Location', 'southwest');
%saveas(figure(3), 'localization_error_hybrid_feedback.png');
export_fig localization_error_hybrid_feedback -pdf 



%% CLASSIFICATION ERROR PLOTS - HYBRID - Different models - SECOND PASS

feedback_classification_legend = {...
    char('top-1 2� feed-foward (Foveal) CaffeNet ');...
    char('top-5 2� feed-foward (Foveal) CaffeNet ');...
    char('top-1 2� feed-foward (Foveal) VGGNet');...
    char('top-5 2� feed-foward (Foveal) VGGNet');...
    char('top-1 2� feed-foward (Foveal) GoogLeNet');...
    char('top-5 2� feed-foward (Foveal) GoogLeNet');...
    };


figure(4)
set(gcf, 'Color', [1,1,1]);
plot(feedback_sigmas,100*hybrid2_top1_classification_error_rate(:,1),'r--o'); 
hold on
plot(feedback_sigmas,100*hybrid2_top5_classification_error_rate(:,1),'r-o');
plot(feedback_sigmas,100*vgg_hybrid2_top1_classification_error_rate(:,1),'g--*'); 
plot(feedback_sigmas,100*vgg_hybrid2_top5_classification_error_rate(:,1),'g-*');
plot(feedback_sigmas,100*google_hybrid2_top1_classification_error_rate(:,1),'b--s'); 
plot(feedback_sigmas,100*google_hybrid2_top5_classification_error_rate(:,1),'b-s');

xlabel('$\sigma_f$','Interpreter','LaTex','FontSize',fontsize);
ylabel('Classification Error (%)','Interpreter','LaTex','FontSize',fontsize);
ylim([0 100])
set(gca,'XTick',[0:20:100],'YTick',[0:20:100], 'FontSize', fontsize);
%set(gca, 'FontSize', fontsize);
%legend(feedback_classification_legend(:),'Location', 'southwest');  % southeast
%saveas(figure(4),'classification_error_hybrid_feedback.png')
export_fig classification_error_hybrid_feedback -pdf





