images_number=100;
gt_folder='../dataset/gt/';
detections_file='../dataset/raw_bbox.txt';


gt=parse_ground_truth(gt_folder,images_number);
detections=parse_detections(detections_file);

%bboxOverlapRatio(bboxA,bboxB)
