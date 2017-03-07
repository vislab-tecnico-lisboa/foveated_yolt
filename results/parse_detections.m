function [sigmas,classes,scores,detections] = parse_detections(detections_file)
detections_ = tdfread(detections_file,';');

sigmas=detections_.sigma;

scores=zeros(length(detections_.score1),5);
scores(:,1)=detections_.score1;
scores(:,2)=detections_.score2;
scores(:,3)=detections_.score3;
scores(:,4)=detections_.score4;
scores(:,5)=detections_.score5;

classes=[];
detections=zeros(length(detections_.x1),5,4);

for i=1:length(detections_.x1)
    classes=[classes; char(detections_.class1(i,:)),...
        char(detections_.class2(i,:)),...
        char(detections_.class3(i,:)),...
        char(detections_.class4(i,:)),...
        char(detections_.class5(i,:))];

    detections(i,1,1)=detections_.x1(i);
    detections(i,1,2)=detections_.y1(i);
    detections(i,1,3)=detections_.w1(i);
    detections(i,1,4)=detections_.h1(i);
    
    detections(i,2,1)=detections_.x2(i);
    detections(i,2,2)=detections_.y2(i);
    detections(i,2,3)=detections_.w2(i);
    detections(i,2,4)=detections_.h2(i);
    
    detections(i,3,1)=detections_.x3(i);
    detections(i,3,2)=detections_.y3(i);
    detections(i,3,3)=detections_.w3(i);
    detections(i,3,4)=detections_.h3(i);
    
    detections(i,4,1)=detections_.x4(i);
    detections(i,4,2)=detections_.y4(i);
    detections(i,4,3)=detections_.w4(i);
    detections(i,4,4)=detections_.h4(i);
    
    detections(i,5,1)=detections_.x5(i);
    detections(i,5,2)=detections_.y5(i);
    detections(i,5,3)=detections_.w5(i);
    detections(i,5,4)=detections_.h5(i);
end
