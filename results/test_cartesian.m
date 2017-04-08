close all
addpath('export_fig');

gt_folder='../dataset/gt/';


cartesian_detection ='../dataset/detections/new/raw_bbox_parse_cartesian_caffe.txt';
% google_cartesian_detection ='../dataset/detections/new/raw_bbox_parse_cartesian_google.txt';
% vgg_cartesian_detection ='../dataset/detections/new/raw_bbox_parse_cartesian_vgg.txt';

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
[cartesian_sigmas,cartesian_threshs,cartesian_classes,cartesian_scores,cartesian_detections]=parse_detections(...
    images_number,...
    cartesian_detection);

% % get detections (YOLO) - FOVEAL - GOOGLE
% [google_cartesian_sigmas,google_cartesian_threshs,google_cartesian_classes,google_cartesian_scores,google_cartesian_detections]=parse_detections(...
%     images_number,...
%     google_cartesian_detection);
% 
% % get detections (YOLO) - FOVEAL - VGG
% [vgg_cartesian_sigmas,vgg_cartesian_threshs,vgg_cartesian_classes,vgg_cartesian_scores,vgg_cartesian_detections]=parse_detections(...
%     images_number,...
%     vgg_cartesian_detection);
% 
%  
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
[cartesian_detection_error_rate] = detection_error_rates(cartesian_sigmas,cartesian_threshs,images_number,cartesian_detections,gt_detections,detections_resolution,top_k,overlap_correct);

% % get detection error rates (YOLO) - FOVEAL - GOOGLE
% [google_cartesian_detection_error_rate] = detection_error_rates(google_cartesian_sigmas,google_cartesian_threshs,images_number,google_cartesian_detections,gt_detections,detections_resolution,top_k,overlap_correct);
% 
% % get detection error rates (YOLO) - FOVEAL - VGG
% [vgg_cartesian_detection_error_rate] = detection_error_rates(vgg_cartesian_sigmas,vgg_cartesian_threshs,images_number,vgg_cartesian_detections,gt_detections,detections_resolution,top_k,overlap_correct);



%% CLASSIFICATION
% get classification error rates (YOLO) - CAFFE - FOVEAL - CAFFE
[cartesian_top1_classification_error_rate, cartesian_top5_classification_error_rate] = classification_error_rates(cartesian_sigmas,cartesian_threshs,images_number,cartesian_classes,gt_classes,top_k);
% 
% % get classification error rates (YOLO) - CAFFE - FOVEAL - GOOGLE
% [google_cartesian_top1_classification_error_rate, google_cartesian_top5_classification_error_rate] = classification_error_rates(google_cartesian_sigmas,google_cartesian_threshs,images_number,google_cartesian_classes,gt_classes,top_k);
% 
% % get classification error rates (YOLO) - CAFFE - FOVEAL - VGG
% [vgg_cartesian_top1_classification_error_rate, vgg_cartesian_top5_classification_error_rate] = classification_error_rates(vgg_cartesian_sigmas,vgg_cartesian_threshs,images_number,vgg_cartesian_classes,gt_classes,top_k);



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
    char('Backward (Cartesian) CaffeNet \sigma = ');...
    char('Backward (Cartesian) CaffeNet \sigma = ');...
    %char('Backward (Cartesian) GoogLeNet \sigma = ');...
    %char('Backward (Cartesian) GoogLeNet \sigma = ');...
    %char('Backward (Cartesian) VGGNet \sigma = ');...
    %char('Backward (Cartesian) VGGNet \sigma = ');...
    };



figure(1)
fontsize=30;
set(gcf, 'Color', [1,1,1]);  % 
plot(threshs,100*cartesian_detection_error_rate(9,:), 'm--o');   
hold on
plot(threshs,100*cartesian_detection_error_rate(11,:), 'm-o');  
%plot(threshs,100*google_cartesian_detection_error_rate(9,:), 'b--s');   
%plot(threshs,100*google_cartesian_detection_error_rate(11,:), 'b-s'); 
%plot(threshs,100*vgg_cartesian_detection_error_rate(9,:), 'r--*');   
%plot(threshs,100*vgg_cartesian_detection_error_rate(11,:), 'r-*'); 
xlabel('$th$','Interpreter','LaTex','FontSize',fontsize);
ylabel('Localization Error (%)','Interpreter','LaTex','FontSize',fontsize);
%xlim([20 100])
ylim([0 100])
set(gca, 'FontSize', 12);
legend('show', 'DislpayName', localization_legend(:) ,'Location', 'southwest');
saveas(figure(1), 'localization_error_cartesian_100.png');
%export_fig localization_error_cartesian_100 -pdf 



%% CLASSIFICATION ERROR PLOTS - FOVEAL - Different models

classification_legend = {...
    char('top-1 feed-foward (Cartesian) CaffeNet ');...
    char('top-5 feed-foward (Cartesian) CaffeNet ');...
    %char('top-1 feed-foward (Cartesian) GoogLeNet');...
    %char('top-5 feed-foward (Cartesian) GoogLeNet');...
    %char('top-1 feed-foward (Cartesian) VGGNet');...
    %char('top-5 feed-foward (Cartesian) VGGNet');...
    };


figure(2)
fontsize=30;
set(gcf, 'Color', [1,1,1]);
plot(cartesian_sigmas,100*cartesian_top1_classification_error_rate(:,1),'m--o'); 
hold on
plot(cartesian_sigmas,100*cartesian_top5_classification_error_rate(:,1),'m-o');
%plot(cartesian_sigmas,100*google_cartesian_top1_classification_error_rate(:,1),'b--s'); 
%plot(cartesian_sigmas,100*google_cartesian_top5_classification_error_rate(:,1),'b-s');
%plot(cartesian_sigmas,100*vgg_cartesian_top1_classification_error_rate(:,1),'r--*'); 
%plot(cartesian_sigmas,100*vgg_cartesian_top5_classification_error_rate(:,1),'r-*');


xlabel('$\sigma$','Interpreter','LaTex','FontSize',fontsize);
ylabel('Classification Error (%)','Interpreter','LaTex','FontSize',fontsize);
ylim([0 100])
set(gca, 'FontSize', 18);
legend(classification_legend(:),'Location', 'southwest');  % southeast
saveas(figure(2),'classification_error_cartesian_100.png')
%export_fig classification_error_cartesian_100 -pdf





