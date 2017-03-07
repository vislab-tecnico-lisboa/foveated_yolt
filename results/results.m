gt_folder='../dataset/gt/';
detections_file='../dataset/detections/raw_bbox_parse.txt';

% parameters
sigmas_number=10;
thresholds_number=10;
images_number=100;
overlap_correct=0.1;

% get ground truth
gt=parse_ground_truth(gt_folder,images_number);

% get detections
[sigmas,classes,scores,detections]=parse_detections(detections_file);

% check overlaps
overlaps=zeros(sigmas_number,thresholds_number, images_number,5);
for s=1:sigmas_number
    for t=1:thresholds_number
        for i=1:images_number
            %check overlaps
            overlaps(s,t,i,1)=bboxOverlapRatio(gt(i,:),reshape(detections(i,1,:),1,4));
            overlaps(s,t,i,2)=bboxOverlapRatio(gt(i,:),reshape(detections(i,2,:),1,4));
            overlaps(s,t,i,3)=bboxOverlapRatio(gt(i,:),reshape(detections(i,3,:),1,4));
            overlaps(s,t,i,4)=bboxOverlapRatio(gt(i,:),reshape(detections(i,4,:),1,4));
            overlaps(s,t,i,5)=bboxOverlapRatio(gt(i,:),reshape(detections(i,5,:),1,4));
        end
    end
end

% consider only the maximum overlap over all 5 detections
overlaps=max(overlaps,[],4);

% evaluate if the detections were good
overlaps(overlaps>=overlap_correct)=1;
overlaps(overlaps<overlap_correct)=0;

% compute detection rate
detection_rate=sum(overlaps,3)/size(overlaps,3);

% plots...