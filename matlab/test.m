close all
addpath('export_fig');

gt_folder='../dataset/gt/';


foveal_detection ='../dataset/detections/new/raw_bbox_parse_foveal_caffe.txt';

% CAFFE
detections_file='../dataset/detections/new/raw_bbox_parse_vale_high_blur_caffe_100.txt';   
feedback_detections_file = '../dataset/detections/new/feedback_detection_parse_vale_high_blur_caffe_100.txt';

% GOOGLE
detections_google_file='../dataset/detections/new/raw_bbox_parse_vale_high_blur_google_100.txt';   
feedback_detections_google_file = '../dataset/detections/new/feedback_detection_parse_vale_high_blur_google_100.txt';

% VGG
% detections_vgg_file='../dataset/detections/new/raw_bbox_parse_vale_high_blur_vggnet_100.txt';   
% feedback_detections_vgg_file = '../dataset/detections/new/feedback_detection_parse_vale_high_blur_vggnet_100.txt';


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


% get detections (YOLO) - FOVEAL
[foveal_sigmas,foveal_threshs,foveal_classes,foveal_scores,foveal_detections]=parse_detections(...
    images_number,...
    foveal_detection);

% get detections (YOLO) - Caffe
[sigmas,threshs,classes,scores,detections]=parse_detections(...
    images_number,...
    detections_file);

% get detections (YOLO) - Google
[sigmas_google,threshs_google,classes_google,scores_google,detections_google]=parse_detections(...
    images_number,...
    detections_google_file);

% % get detections (YOLO) - VGG
% [sigmas_vgg,threshs_vgg,classes_vgg,scores_vgg,detections_vgg]=parse_detections(...
%     images_number,...
%     detections_vgg_file);


%get feedback detections (YOLT-foveation) - Caffe
[feedback_sigmas,feedback_threshs,feedback_classes,feedback_scores,rank_feedback_classes,feedback_detections]=feedback_parse_detections2(...
     images_number,...
     feedback_detections_file);

 
%get feedback detections (YOLT-foveation) - Google
[feedback_sigmas_google,feedback_threshs_google,feedback_classes_google,feedback_scores_google,rank_feedback_classes_google,feedback_detections_google]=feedback_parse_detections2(...
     images_number,...
     feedback_detections_google_file);
 
% %get feedback detections (YOLT-foveation) - VGG
% [feedback_sigmas_vgg,feedback_threshs_vgg,feedback_classes_vgg,feedback_scores_vgg,rank_feedback_classes_vgg,feedback_detections_vgg]=feedback_parse_detections2(...
%      images_number,...
%      feedback_detections_vgg_file);
 
 
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

% get detection error rates (YOLO) - FOVEAK
[foveal_detection_error_rate] = detection_error_rates(foveal_sigmas,foveal_threshs,images_number,foveal_detections,gt_detections,detections_resolution,top_k,overlap_correct);


% get detection error rates (YOLO) - Caffe
[detection_error_rate] = detection_error_rates(sigmas,threshs,images_number,detections,gt_detections,detections_resolution,top_k,overlap_correct);

% get detection error rates (YOLO) - Google
[detection_google_error_rate] = detection_error_rates(sigmas_google,threshs_google,images_number,detections_google,gt_detections,detections_resolution,top_k,overlap_correct);

% get detection error rates (YOLO) - VGG
%[detection_vgg_error_rate] = detection_error_rates(sigmas_vgg,threshs_vgg,images_number,detections_vgg,gt_detections,detections_resolution,top_k,overlap_correct);




% get detection foveate error rate - Caffe - Feedback
[detection_foveate_error_rate] = detection_error_rates(feedback_sigmas,feedback_threshs,images_number,feedback_detections,gt_detections,detections_resolution,top_k,overlap_correct);

% get detection foveate error rate - Google - Feedback
[detection_foveate_google_error_rate] = detection_error_rates(feedback_sigmas_google,feedback_threshs_google,images_number,feedback_detections_google,gt_detections,detections_resolution,top_k,overlap_correct);

% get detection foveate error rate - VGG - Feedback
%[detection_foveate_vgg_error_rate] = detection_error_rates(feedback_sigmas_vgg,feedback_threshs_vgg,images_number,feedback_detections_vgg,gt_detections,detections_resolution,top_k,overlap_correct);

%% CLASSIFICATION
% get classification error rates (YOLO)
%[top1_classification_error_rate, top5_classification_error_rate] = classification_error_rates(sigmas,threshs,images_number,classes,gt_classes,top_k);

% get feedback classification error rates (YOLT foveation)
%[top1_feedback_classification_error_rate, top5_feedback_classification_error_rate] = classification_error_rates(feedback_sigmas,feedback_threshs,images_number,gt_classes,top_k);




%% detection error plots - CAFFE - 1 passagem - FOVEAL

% fix one sigma and plot all saliency thresholds
sigma_index=1;

%sigmas_leg = [1 20 40 60 80 100];
%plot_detection = [detection_error_rate(1,:); detection_error_rate(3,:); detection_error_rate(5,:); detection_error_rate(7,:); detection_error_rate(9,:); detection_error_rate(11,:)];
legend_sigma = {};
% for i=1:length(sigmas_leg)
%     legend_sigma = [legend_sigma, strcat('\sigma=', num2str(sigmas_leg(i))) ];
% end

% legend_sigma = {};
% for i=1:length(sigmas)
%     legend_sigma = [legend_sigma, strcat('\sigma=', num2str(sigmas(i))) ];
% end

figure(1)
fontsize=30;
set(gcf, 'Color', [1,1,1]);  % 
plot(threshs,100*foveal_detection_error_rate(:,:));   % detection_error_rate(:,:)
xlabel('$th$','Interpreter','LaTex','FontSize',fontsize);
ylabel('Localization Error (%)','Interpreter','LaTex','FontSize',fontsize);
%xlim([20 100])
ylim([0 100])
set(gca, 'FontSize', 20);
%legend('show', 'DislpayName', legend_sigma(:) ,'Location', 'southwest');
%saveas(figure(3),'localization_error_threshold_crop_caffenet_5000.pdf')
%saveas(figure(1), 'vale_first_pass_caffe.png');
%export_fig localization_error_th -pdf 









%% detection error plots - CAFFE - 1 passagem
% 
% % fix one sigma and plot all saliency thresholds
% sigma_index=1;
% 
% %sigmas_leg = [1 20 40 60 80 100];
% %plot_detection = [detection_error_rate(1,:); detection_error_rate(3,:); detection_error_rate(5,:); detection_error_rate(7,:); detection_error_rate(9,:); detection_error_rate(11,:)];
% legend_sigma = {};
% for i=1:length(sigmas_leg)
%     legend_sigma = [legend_sigma, strcat('\sigma=', num2str(sigmas_leg(i))) ];
% end
% 
% % legend_sigma = {};
% % for i=1:length(sigmas)
% %     legend_sigma = [legend_sigma, strcat('\sigma=', num2str(sigmas(i))) ];
% % end
% 
% figure(1)
% fontsize=30;
% set(gcf, 'Color', [1,1,1]);  % 
% plot(threshs,100*detection_error_rate(9,:), 'b-s');   % detection_error_rate(:,:)
% xlabel('$th$','Interpreter','LaTex','FontSize',fontsize);
% ylabel('Localization Error (%)','Interpreter','LaTex','FontSize',fontsize);
% %xlim([20 100])
% ylim([0 100])
% set(gca, 'FontSize', 20);
% %legend('show', 'DislpayName', legend_sigma(:) ,'Location', 'southwest');
% %saveas(figure(3),'localization_error_threshold_crop_caffenet_5000.pdf')
% saveas(figure(1), 'vale_first_pass_caffe.png');
% %export_fig localization_error_th -pdf 


%% detection error plot - Caffe - Feedback
% sigma_index=1;
% 
% sigmas_leg = [1 20 40 60 80 100];
% plot_feedback_detection = [detection_foveate_error_rate(1,:); detection_foveate_error_rate(3,:); detection_foveate_error_rate(5,:); detection_foveate_error_rate(7,:); detection_foveate_error_rate(9,:); detection_foveate_error_rate(11,:)];
% legend_sigma = {};
% for i=1:length(sigmas_leg)
%     legend_sigma = [legend_sigma, strcat('\sigma=', num2str(sigmas_leg(i))) ];
% end
% 
% % legend_sigma = {};
% % for i=1:length(sigmas)
% %     legend_sigma = [legend_sigma, strcat('\sigma=', num2str(sigmas(i))) ];
% % end
% 
% figure(2)
% fontsize=30;
% set(gcf, 'Color', [1,1,1]);  % 
% plot(threshs,100*plot_feedback_detection(:,:));   % detection_error_rate(:,:)
% xlabel('$th$','Interpreter','LaTex','FontSize',fontsize);
% ylabel('Localization Error (%)','Interpreter','LaTex','FontSize',fontsize);
% %xlim([20 100])
% ylim([0 100])
% set(gca, 'FontSize', 20);
% legend('show', 'DislpayName', legend_sigma(:) ,'Location', 'southwest');
% %saveas(figure(3),'localization_error_threshold_crop_caffenet_5000.pdf')
% saveas(figure(2), 'vale_feedback_fovea_caffe.png');
% %export_fig localization_error_th -pdf 




%% Plot junto - 1 pass + feedback fovea - CAFFENET

sigma_index=1;

sigmas_leg = [40 60 80 100];
plot_feedback_detection = [detection_foveate_error_rate(5,:); detection_foveate_error_rate(7,:); detection_foveate_error_rate(9,:); detection_foveate_error_rate(11,:)];
legend_sigma = {};
for i=1:length(sigmas_leg)
    legend_sigma = [legend_sigma, strcat('\sigma=', num2str(sigmas_leg(i))) ];
end


localization_mix_leg = {...
    char('Feed-forward ');...
    %char('Fovea \sigma=1 ');...
    %char('Fovea \sigma=20 ');...
    char('Fovea \sigma=40 ');...
    char('Fovea \sigma=60 ');... 
    char('Fovea \sigma=80 ');...
    char('Fovea \sigma=100 ');...
    %char('Crop ');...
    };

figure(3)
fontsize=30;
set(gcf, 'Color', [1,1,1]);  % 
plot(threshs,100*detection_error_rate(9,:), 'k-s');   % detection_error_rate(:,:)
hold on
plot(threshs,100*plot_feedback_detection(:,:));   % detection_error_rate(:,:)
xlabel('$th$','Interpreter','LaTex','FontSize',fontsize);
ylabel('Localization Error (%)','Interpreter','LaTex','FontSize',fontsize);
%xlim([20 100])
ylim([0 100])
set(gca, 'FontSize', 20);
legend('show', 'DislpayName', localization_mix_leg(:) ,'Location', 'southwest');  %southeast
saveas(figure(3), 'vale_mix_caffe.png');






%% Plot junto - 1 pass + feedback fovea - GOOGLENET

sigma_index=1;

sigmas_leg = [40 60 80 100];
plot_google_feedback_detection = [detection_foveate_google_error_rate(5,:); detection_foveate_google_error_rate(7,:); detection_foveate_google_error_rate(9,:); detection_foveate_google_error_rate(11,:)];
legend_sigma = {};
for i=1:length(sigmas_leg)
    legend_sigma = [legend_sigma, strcat('\sigma=', num2str(sigmas_leg(i))) ];
end


localization_mix_leg = {...
    char('Feed-forward ');...
   % char('Fovea \sigma=1 ');...
   % char('Fovea \sigma=20 ');...
    char('Fovea \sigma=40 ');...
    char('Fovea \sigma=60 ');... 
    char('Fovea \sigma=80 ');...
    char('Fovea \sigma=100 ');...
    %char('Crop ');...
    };

figure(4)
fontsize=30;
set(gcf, 'Color', [1,1,1]);  % 
plot(threshs,100*detection_google_error_rate(10,:), 'k-s');   % 1 passagem
hold on
plot(threshs,100*plot_google_feedback_detection(:,:));   % 2 passagem - foveada
xlabel('$th$','Interpreter','LaTex','FontSize',fontsize);
ylabel('Localization Error (%)','Interpreter','LaTex','FontSize',fontsize);
%xlim([20 100])
ylim([0 100])
set(gca, 'FontSize', 20);
legend('show', 'DislpayName', localization_mix_leg(:) ,'Location', 'southwest');  %southeast
saveas(figure(4), 'vale_mix_google.png');





%% detection error plots - GOOGLE

% % fix one sigma and plot all saliency thresholds
% sigma_index=1;
% 
% sigmas_leg = [1 20 40 60 80 100];
% plot_google_detection = [detection_google_error_rate(1,:); detection_google_error_rate(3,:); detection_google_error_rate(5,:); detection_google_error_rate(7,:); detection_google_error_rate(9,:); detection_google_error_rate(11,:)];
% legend_sigma = {};
% for i=1:length(sigmas_leg)
%     legend_sigma = [legend_sigma, strcat('\sigma=', num2str(sigmas_leg(i))) ];
% end
% 
% % legend_sigma = {};
% % for i=1:length(sigmas)
% %     legend_sigma = [legend_sigma, strcat('\sigma=', num2str(sigmas(i))) ];
% % end
% 
% figure(5)
% fontsize=30;
% set(gcf, 'Color', [1,1,1]);  % 
% plot(threshs,100*detection_google_error_rate(:,:));   % detection_google_error_rate(:,:)
% xlabel('$th$','Interpreter','LaTex','FontSize',fontsize);
% ylabel('Localization Error (%)','Interpreter','LaTex','FontSize',fontsize);
% %xlim([20 100])
% ylim([0 100])
% set(gca, 'FontSize', 20);
% legend('show', 'DislpayName', legend_sigma(:) ,'Location', 'southwest');
% %saveas(figure(3),'localization_error_threshold_crop_caffenet_5000.pdf')
% saveas(figure(5), 'vale_google.png');
% %export_fig localization_error_th -pdf 


