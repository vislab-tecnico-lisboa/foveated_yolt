gt_folder='../dataset/gt/';
detections_file='../dataset/detections/raw_bbox_parse.txt';

% parameters
sigmas_number=10;
thresholds_number=10;
images_number=100;
overlap_correct=0.1;
top_k=5;
% get ground truth
gt=parse_ground_truth(gt_folder,images_number);

% get detections
[sigmas,classes,scores,detections]=parse_detections(detections_file);

% check overlaps
overlaps=zeros(sigmas_number,thresholds_number, images_number,top_k);
overlaps=[];
for s=1:sigmas_number
    for t=1:thresholds_number
        for i=1:images_number
            %check overlaps for each ground truth bounding box
            overlaps(s,t,i,1).overlap=zeros(size(gt(i).bboxes,1),1);
            for g=1:size(gt(i).bboxes,1)
                gt_bbox=gt(i).bboxes(g,:);
                
                % for each detection (of the 5)
                for j=1:top_k
                    overlaps(s,t,i,j).overlap(g)=bboxOverlapRatio(gt_bbox,reshape(detections(i,j,:),1,4));
                end
            end
        end
    end
end

% consider only the maximum overlap over ground truths
max_overlap=zeros(sigmas_number,thresholds_number, images_number,top_k);
for s=1:sigmas_number
    for t=1:thresholds_number
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
detection_rate=sum(max_overlap,3)/size(max_overlap,3);

%% plots

% fix one threshold and plot all sigmas
thresh_index=1;

figure(1)
fontsize=15;
set(gcf, 'Color', [1,1,1]);
plot(detection_rate(:,thresh_index))
xlabel('$\sigma$','Interpreter','LaTex','FontSize',fontsize);
ylabel('detection rate','Interpreter','LaTex','FontSize',fontsize);


% fix one sigma and plot all saliency thresholds
sigma_index=1;

figure(2)
fontsize=15;
set(gcf, 'Color', [1,1,1]);
plot(detection_rate(sigma_index,:))
xlabel('$th$','Interpreter','LaTex','FontSize',fontsize);
ylabel('detection rate','Interpreter','LaTex','FontSize',fontsize);

%export_fig error_results -pdf


