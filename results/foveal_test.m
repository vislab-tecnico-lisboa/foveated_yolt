close all
addpath('export_fig');

gt_folder='../dataset/gt/';


foveal_detection ='../dataset/detections/new/raw_bbox_parse_foveal_caffe.txt';
google_foveal_detection ='../dataset/detections/new/raw_bbox_parse_foveal_google.txt';
vgg_foveal_detection ='../dataset/detections/new/raw_bbox_parse_foveal_vgg.txt';

classifications_file='../files/ground_truth_labels_ilsvrc12.txt';
images_folder='../dataset/images/';

% set to true to check detections
view_detections=false;

% parameters
detections_resolution=227;
images_number=100; 
overlap_correct=0.5;
top_k=5;


%% DETECTIONS

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

% get detection error rates (YOLO) - FOVEAL - CAFFE
[foveal_detection_error_rate] = detection_error_rates(foveal_sigmas,foveal_threshs,images_number,foveal_detections,gt_detections,detections_resolution,top_k,overlap_correct);

% get detection error rates (YOLO) - FOVEAL - GOOGLE
[google_foveal_detection_error_rate] = detection_error_rates(google_foveal_sigmas,google_foveal_threshs,images_number,google_foveal_detections,gt_detections,detections_resolution,top_k,overlap_correct);

% get detection error rates (YOLO) - FOVEAL - VGG
[vgg_foveal_detection_error_rate] = detection_error_rates(vgg_foveal_sigmas,vgg_foveal_threshs,images_number,vgg_foveal_detections,gt_detections,detections_resolution,top_k,overlap_correct);



%% CLASSIFICATION
% get classification error rates (YOLO) - CAFFE - FOVEAL - CAFFE
[foveal_top1_classification_error_rate, foveal_top5_classification_error_rate] = classification_error_rates(foveal_sigmas,foveal_threshs,images_number,foveal_classes,gt_classes,top_k);

% get classification error rates (YOLO) - CAFFE - FOVEAL - GOOGLE
[google_foveal_top1_classification_error_rate, google_foveal_top5_classification_error_rate] = classification_error_rates(google_foveal_sigmas,google_foveal_threshs,images_number,google_foveal_classes,gt_classes,top_k);

% get classification error rates (YOLO) - CAFFE - FOVEAL - VGG
[vgg_foveal_top1_classification_error_rate, vgg_foveal_top5_classification_error_rate] = classification_error_rates(vgg_foveal_sigmas,vgg_foveal_threshs,images_number,vgg_foveal_classes,gt_classes,top_k);



%% detection error plots - CAFFE - 1 passagem - FOVEAL

% fix one sigma and plot all saliency thresholds
sigma_index=1;

% sigmas_leg = [1 20 40 60 80 100];
% plot_detection = [foveal_detection_error_rate(1,:); foveal_detection_error_rate(3,:); foveal_detection_error_rate(5,:); foveal_detection_error_rate(7,:); foveal_detection_error_rate(9,:); foveal_detection_error_rate(11,:)];
% legend_sigma = {};
% for i=1:length(sigmas_leg)
%     legend_sigma = [legend_sigma, strcat('\sigma=', num2str(sigmas_leg(i))) ];
% end

localization_legend = {...
    char('Backward (Foveal) CaffeNet \sigma = 80');...
    char('Backward (Foveal) CaffeNet \sigma = 100');...
    char('Backward (Foveal) GoogLeNet \sigma = 80');...
    char('Backward (Foveal) GoogLeNet \sigma = 100');...
    char('Backward (Foveal) VGGNet \sigma = 80');...
    char('Backward (Foveal) VGGNet \sigma = 100');...
    };



figure(1)
fontsize=30;
set(gcf, 'Color', [1,1,1]);  % 
plot(threshs,100*foveal_detection_error_rate(9,:), 'm--o');   
hold on
plot(threshs,100*foveal_detection_error_rate(11,:), 'm-o');  
plot(threshs,100*google_foveal_detection_error_rate(9,:), 'b--s');   
plot(threshs,100*google_foveal_detection_error_rate(11,:), 'b-s'); 
plot(threshs,100*vgg_foveal_detection_error_rate(9,:), 'r--*');   
plot(threshs,100*vgg_foveal_detection_error_rate(11,:), 'r-*'); 
xlabel('$th$','Interpreter','LaTex','FontSize',fontsize);
ylabel('Localization Error (%)','Interpreter','LaTex','FontSize',fontsize);
%xlim([20 100])
ylim([0 100])
set(gca, 'FontSize', 20);
legend('show', 'DislpayName', localization_legend(:) ,'Location', 'northwest');
saveas(figure(1), 'localization_error_foveal_100.png');
%export_fig localization_error_foveal_100 -pdf 



%% CLASSIFICATION ERROR PLOTS - FOVEAL - Different models

classification_legend = {...
    char('top-1 feed-foward (Foveal) CaffeNet ');...
    char('top-5 feed-foward (Foveal) CaffeNet ');...
    char('top-1 feed-foward (Foveal) GoogLeNet');...
    char('top-5 feed-foward (Foveal) GoogLeNet');...
    char('top-1 feed-foward (Foveal) VGGNet');...
    char('top-5 feed-foward (Foveal) VGGNet');...
    };


figure(2)
fontsize=30;
set(gcf, 'Color', [1,1,1]);
plot(foveal_sigmas,100*foveal_top1_classification_error_rate(:,1),'m--o'); 
hold on
plot(foveal_sigmas,100*foveal_top5_classification_error_rate(:,1),'m-o');
plot(foveal_sigmas,100*google_foveal_top1_classification_error_rate(:,1),'b--s'); 
plot(foveal_sigmas,100*google_foveal_top5_classification_error_rate(:,1),'b-s');
plot(foveal_sigmas,100*vgg_foveal_top1_classification_error_rate(:,1),'r--*'); 
plot(foveal_sigmas,100*vgg_foveal_top5_classification_error_rate(:,1),'r-*');


xlabel('$\sigma$','Interpreter','LaTex','FontSize',fontsize);
ylabel('Classification Error (%)','Interpreter','LaTex','FontSize',fontsize);
ylim([0 100])
set(gca, 'FontSize', 20);
legend(classification_legend(:),'Location', 'southwest');  % southeast
saveas(figure(2),'classification_error_foveal_100.png')
%export_fig classification_error_foveal_100 -pdf





