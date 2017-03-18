close all
addpath('export_fig');

gt_folder='../dataset/gt/';
detections_file='../dataset/detections/raw_bbox_parse_vggnet_5000.txt';
classifications_file='../files/ground_truth_labels_ilsvrc12.txt';
images_folder='../dataset/images/';

% set to true to check detections
view_detections=false;

% parameters
detections_resolution=227;
images_number=100; 
overlap_correct=0.5;
top_k=5;

% get ground truth
[gt_detections, gt_classes]=parse_ground_truth(gt_folder,classifications_file,images_number);

% get detections
[sigmas,threshs,classes,scores,detections]=parse_detections(...
    images_number,...
    detections_file);

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

% get detection error rates
[detection_error_rate] = detection_error_rates(sigmas,threshs,images_number,detections,gt_detections,detections_resolution,top_k,overlap_correct);

% get classification error rates
[top1_classification_error_rate, top5_classification_error_rate] = classification_error_rates(sigmas,threshs,images_number,classes,gt_classes,top_k);



%% detection error plots

% fix one threshold and plot all sigmas
thresh_index=1;

legend_thres = {};
for i=1:length(threshs)
    legend_thres = [legend_thres, strcat('th=', num2str(threshs(i))) ];
end

figure(2)
fontsize=15;
set(gcf, 'Color', [1,1,1]);
plot(sigmas,100*detection_error_rate(:,:)) %15
xlabel('$\sigma$','Interpreter','LaTex','FontSize',fontsize);
ylabel('Localization Error (%)','Interpreter','LaTex','FontSize',fontsize);
ylim([0 100])
legend('show', 'DislpayName', legend_thres(:) ,'Location', 'bestoutside');
saveas(figure(2),'localization_error_sigma_vggnet_100.pdf')
%export_fig localization_error_sigma -pdf

% fix one sigma and plot all saliency thresholds
sigma_index=1;

legend_sigma = {};
for i=1:length(sigmas)
    legend_sigma = [legend_sigma, strcat('\sigma=', num2str(sigmas(i))) ];
end

figure(3)
fontsize=15;
set(gcf, 'Color', [1,1,1]);  % set background color to white
plot(threshs,100*detection_error_rate(:,:))
xlabel('$th$','Interpreter','LaTex','FontSize',fontsize);
ylabel('Localization Error (%)','Interpreter','LaTex','FontSize',fontsize);
xlim([20 100])
ylim([0 100])
legend('show', 'DislpayName', legend_sigma(:) ,'Location', 'bestoutside');
saveas(figure(3),'localization_error_threshold_vggnet_100.pdf')

%export_fig localization_error_th -pdf

% %% classification (top 1) error plots
% 
% % fix one threshold and plot all sigmas
% thresh_index=1;
% 
% legend_thres = {};
% for i=1:length(threshs)
%     legend_thres = [legend_thres, strcat('th=', num2str(threshs(i))) ];
% end
% 
% figure(4)
% fontsize=15;
% set(gcf, 'Color', [1,1,1]);
% plot(sigmas,100*top1_classification_error_rate(:,:)) %15
% xlabel('$\sigma$','Interpreter','LaTex','FontSize',fontsize);
% ylabel('Classification Error (top 1) (%)','Interpreter','LaTex','FontSize',fontsize);
% ylim([0 100])
% legend('show', 'DislpayName', legend_thres(:) ,'Location', 'bestoutside');
% saveas(figure(2),'classification_top1_error_sigma_caffenet_100.pdf')
% %export_fig localization_error_sigma -pdf
% 
% % fix one sigma and plot all saliency thresholds
% sigma_index=1;
% 
% legend_sigma = {};
% for i=1:length(sigmas)
%     legend_sigma = [legend_sigma, strcat('\sigma=', num2str(sigmas(i))) ];
% end
% 
% figure(5)
% fontsize=15;
% set(gcf, 'Color', [1,1,1]);  % set background color to white
% plot(threshs,100*top1_classification_error_rate(:,:))
% 
% xlabel('$th$','Interpreter','LaTex','FontSize',fontsize);
% ylabel('Classification Error (top 1) (%)','Interpreter','LaTex','FontSize',fontsize);
% xlim([20 100])
% ylim([0 100])
% legend('show', 'DislpayName', legend_sigma(:) ,'Location', 'bestoutside');
% saveas(figure(3),'classification_top1_error_threshold_caffenet_100.pdf')
% 
% %% classification (top 5) error plots
% 
% % fix one threshold and plot all sigmas
% thresh_index=1;
% 
% legend_thres = {};
% for i=1:length(threshs)
%     legend_thres = [legend_thres, strcat('th=', num2str(threshs(i))) ];
% end
% 
% figure(6)
% fontsize=15;
% set(gcf, 'Color', [1,1,1]);
% plot(sigmas,100*top5_classification_error_rate(:,:)) %15
% xlabel('$\sigma$','Interpreter','LaTex','FontSize',fontsize);
% ylabel('Classification Error (top 5) (%)','Interpreter','LaTex','FontSize',fontsize);
% ylim([0 100])
% legend('show', 'DislpayName', legend_thres(:) ,'Location', 'bestoutside');
% saveas(figure(2),'classification_top5_error_sigma_caffenet_100.pdf')
% %export_fig localization_error_sigma -pdf
% 
% % fix one sigma and plot all saliency thresholds
% sigma_index=1;
% 
% legend_sigma = {};
% for i=1:length(sigmas)
%     legend_sigma = [legend_sigma, strcat('\sigma=', num2str(sigmas(i))) ];
% end
% 
% figure(7)
% fontsize=15;
% set(gcf, 'Color', [1,1,1]);  % set background color to white
% plot(threshs,100*top5_classification_error_rate(:,:))
% 
% xlabel('$th$','Interpreter','LaTex','FontSize',fontsize);
% ylabel('Classification Error (top 5) (%)','Interpreter','LaTex','FontSize',fontsize);
% xlim([20 100])
% ylim([0 100])
% legend('show', 'DislpayName', legend_sigma(:) ,'Location', 'bestoutside');
% saveas(figure(3),'classification_top5_error_threshold_caffenet_100.pdf')

