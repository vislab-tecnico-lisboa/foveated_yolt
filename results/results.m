close all
addpath('export_fig');

gt_folder='../dataset/gt/';
detections_file='../dataset/detections/raw_bbox_parse_5000.txt';
classifications_file='../files/ground_truth_labels_ilsvrc12.txt';
images_folder='../dataset/images/';

% set to true to check detections
view_detections=false;

% parameters
detections_resolution=227;
images_number=100;  %100;  %5000;
overlap_correct=0.5;
top_k=5;

% get ground truth
gt=parse_ground_truth(gt_folder,images_number);

% get detections
[sigmas,threshs,classes,scores,detections]=parse_detections(...
    images_number,...
    detections_file);

% view images
if view_detections
    for i=1:images_number
        figure(i)
        imshow(strcat(images_folder,gt(i).filename))
        hold on
        for g=1:size(gt(i).bboxes,1)
            % gt bbox
            gt_bbox=gt(i).bboxes(g,:);
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

% check overlaps
overlaps=[];
for s=1:length(sigmas)
    for t=1:length(threshs)
        for i=1:images_number
            gt_size=gt(i).size;
            aspect_ratio_x=gt_size(1)/detections_resolution;
            aspect_ratio_y=gt_size(2)/detections_resolution;
            % for each detection (of the 5)
            for j=1:top_k
                % scale detections
                detection=reshape(detections(s,t,i,j,:),1,4);
                detection(1)=detection(1)*aspect_ratio_x;
                detection(2)=detection(2)*aspect_ratio_y;
                detection(3)=detection(3)*aspect_ratio_x;
                detection(4)=detection(4)*aspect_ratio_y;
                % check overlaps for each ground truth bounding box
                overlaps(s,t,i,j).overlap=zeros(size(gt(i).bboxes,1),1);
                
                for g=1:size(gt(i).bboxes,1)
                    % gt bbox
                    gt_bbox=gt(i).bboxes(g,:);
                    
                    overlaps(s,t,i,j).overlap(g)=bboxOverlapRatio(gt_bbox,detection);
                    
                end
            end
        end
    end
end

% consider only the maximum overlap over ground truths
max_overlap=zeros(length(sigmas),length(threshs), images_number,top_k);
for s=1:length(sigmas)
    for t=1:length(threshs)
        for i=1:images_number
            for j=1:top_k
                max_overlap(s,t,i,j)=max(overlaps(s,t,i,j).overlap);
            end
        end
    end
end

% consider only the maximum overlap over all 5 detections
max_overlap=max(max_overlap,[],4);

% evaluate if the detections were good
max_overlap(max_overlap>=overlap_correct)=1;
max_overlap(max_overlap<overlap_correct)=0;

% compute detection rate
error_rate=(size(max_overlap,3)-sum(max_overlap,3))/size(max_overlap,3);

%% plots

% fix one threshold and plot all sigmas
thresh_index=1;

legend_thres = {};
for i=1:length(threshs)
    legend_thres = [legend_thres, strcat('th=', num2str(threshs(i))) ];
end

figure(2)
fontsize=15;
set(gcf, 'Color', [1,1,1]);
plot(sigmas,100*error_rate(:,:)) %15
xlabel('$\sigma$','Interpreter','LaTex','FontSize',fontsize);
ylabel('Localization Error (%)','Interpreter','LaTex','FontSize',fontsize);
ylim([0 100])
legend('show', 'DislpayName', legend_thres(:) ,'Location', 'bestoutside');
saveas(figure(2),'localization_error_sigma_caffenet5000.pdf')
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
plot(threshs,100*error_rate(:,:))
xlabel('$th$','Interpreter','LaTex','FontSize',fontsize);
ylabel('Localization Error (%)','Interpreter','LaTex','FontSize',fontsize);
xlim([20 100])
ylim([0 100])
legend('show', 'DislpayName', legend_sigma(:) ,'Location', 'bestoutside');
saveas(figure(3),'localization_error_threshold_caffenet5000.pdf')

%export_fig localization_error_th -pdf


