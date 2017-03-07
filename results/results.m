images_number=100;
gt_folder='../dataset/gt/';
detections_file='../dataset/detections/raw_bbox_parse.txt';


gt=parse_ground_truth(gt_folder,images_number);
[sigmas,class1,class2,class3,class4,class5,score1,score2,score3,score4,score5,detections1,detections2,detections3,detections4,detections5]=parse_detections(detections_file);

%bboxOverlapRatio(bboxA,bboxB)
