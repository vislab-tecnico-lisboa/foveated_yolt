close all
addpath('export_fig');

gt_folder='../dataset/gt/';
detections_file='../dataset/detections/raw_bbox_parse.txt';
feedback_detections_file = '../dataset/detections/feedback_detection_parse_caffenet_100.txt';
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

% get feedback detections
[feedback_sigmas,feedback_threshs,feedback_classes,feedback_scores]=feedback_parse_detections(...
    images_number,...
    feedback_detections_file);


rank_feedback_classes = cell(length(feedback_sigmas),length(feedback_threshs),images_number,5);

for s=1:length(feedback_sigmas)
    for t=1:length(feedback_threshs)
        for im=1:images_number

            % Rank 25 predicted class labels to top5 final solution
            [rank_feedback_scores, rank_score_index] = sort(feedback_scores(im,:), 'descend');
            
            rank_feedback_classes{s,t,im,1}=char(feedback_classes{s,t,im,rank_score_index(1)});
            rank_feedback_classes{s,t,im,2}=char(feedback_classes{s,t,im,rank_score_index(2)});
            rank_feedback_classes{s,t,im,3}=char(feedback_classes{s,t,im,rank_score_index(3)});
            rank_feedback_classes{s,t,im,4}=char(feedback_classes{s,t,im,rank_score_index(4)});
            rank_feedback_classes{s,t,im,5}=char(feedback_classes{s,t,im,rank_score_index(5)});
        end
    end
end


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

% get classification error rates (YOLO)
[top1_classification_error_rate, top5_classification_error_rate] = classification_error_rates(sigmas,threshs,images_number,classes,gt_classes,top_k);

% get feedback classification error rates (YOLT)
[top1_feedback_classification_error_rate, top5_feedback_classification_error_rate] = classification_error_rates(feedback_sigmas,feedback_threshs,images_number,rank_feedback_classes,gt_classes,top_k);

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
plot(sigmas,100*detection_error_rate(:,:)) 
xlabel('$\sigma$','Interpreter','LaTex','FontSize',fontsize);
ylabel('Localization Error (%)','Interpreter','LaTex','FontSize',fontsize);
ylim([0 100])
legend('show', 'DislpayName', legend_thres(:) ,'Location', 'bestoutside');
saveas(figure(2),'localization_error_sigma_100.pdf')
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
saveas(figure(3),'localization_error_threshold_100.pdf')

%export_fig localization_error_th -pdf 



%% classification (top 1) error plots

% fix one threshold and plot all sigmas
thresh_index=1;

legend_thres = {};
for i=1:length(feedback_threshs)
    legend_thres = [legend_thres, strcat('th=', num2str(feedback_threshs(i))) ];
end

figure(4)
fontsize=15;
set(gcf, 'Color', [1,1,1]);
plot(feedback_sigmas,100*top1_feedback_classification_error_rate(:,:)) %15
xlabel('$\sigma$','Interpreter','LaTex','FontSize',fontsize);
ylabel('Classification Error (top 1) (%)','Interpreter','LaTex','FontSize',fontsize);
ylim([0 100])
legend('show', 'DislpayName', legend_thres(:) ,'Location', 'bestoutside');
saveas(figure(4),'classification_top1_error_sigma_caffenet_100.pdf')
%export_fig localization_error_sigma -pdf

% fix one sigma and plot all saliency thresholds
sigma_index=1;

legend_sigma = {};
for i=1:length(feedback_sigmas)
    legend_sigma = [legend_sigma, strcat('\sigma=', num2str(feedback_sigmas(i))) ];
end

figure(5)
fontsize=15;
set(gcf, 'Color', [1,1,1]);  % set background color to white
plot(feedback_threshs,100*top1_feedback_classification_error_rate(:,:))

xlabel('$th$','Interpreter','LaTex','FontSize',fontsize);
ylabel('Classification Error (top 1) (%)','Interpreter','LaTex','FontSize',fontsize);
xlim([20 100])
ylim([0 100])
legend('show', 'DislpayName', legend_sigma(:) ,'Location', 'bestoutside');
saveas(figure(5),'classification_top1_error_threshold_caffenet_100.pdf')

%% classification (top 5) error plots

% fix one threshold and plot all sigmas
thresh_index=1;

legend_thres = {};
for i=1:length(feedback_threshs)
    legend_thres = [legend_thres, strcat('th=', num2str(feedback_threshs(i))) ];
end

figure(6)
fontsize=15;
set(gcf, 'Color', [1,1,1]);
plot(feedback_sigmas,100*top5_feedback_classification_error_rate(:,:)) %15
xlabel('$\sigma$','Interpreter','LaTex','FontSize',fontsize);
ylabel('Classification Error (top 5) (%)','Interpreter','LaTex','FontSize',fontsize);
ylim([0 100])
legend('show', 'DislpayName', legend_thres(:) ,'Location', 'bestoutside');
saveas(figure(6),'classification_top5_error_sigma_caffenet_100.pdf')
%export_fig localization_error_sigma -pdf

% fix one sigma and plot all saliency thresholds
sigma_index=1;

legend_sigma = {};
for i=1:length(feedback_sigmas)
    legend_sigma = [legend_sigma, strcat('\sigma=', num2str(feedback_sigmas(i))) ];
end

figure(7)
fontsize=15;
set(gcf, 'Color', [1,1,1]);  % set background color to white
plot(feedback_threshs,100*top5_feedback_classification_error_rate(:,:))

xlabel('$th$','Interpreter','LaTex','FontSize',fontsize);
ylabel('Classification Error (top 5) (%)','Interpreter','LaTex','FontSize',fontsize);
xlim([20 100])
ylim([0 100])
legend('show', 'DislpayName', legend_sigma(:) ,'Location', 'bestoutside');
saveas(figure(7),'classification_top5_error_threshold_caffenet_100.pdf')

