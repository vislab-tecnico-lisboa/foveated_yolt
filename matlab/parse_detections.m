function [sigmas,thres,classes,scores,detections] = parse_detections(images_number,detections_file)
detections_ = tdfread(detections_file,';');

sigmas=unique(detections_.sigma);
thres=unique(detections_.thres);
scores=zeros(length(detections_.score1),5);
scores(:,1)=detections_.score1;
scores(:,2)=detections_.score2;
scores(:,3)=detections_.score3;
scores(:,4)=detections_.score4;
scores(:,5)=detections_.score5;

classes=cell(length(sigmas),length(thres),images_number,5);
detections=zeros(length(sigmas),length(thres),images_number,5,4);

total_images=length(detections_.sigma)/(length(sigmas)*length(thres));

for s=1:length(sigmas)
    for t=1:length(thres)
        for im=1:images_number
            i=im+(t-1)*total_images +(s-1)*total_images*length(thres);
            classes{s,t,im,1}=char(detections_.class1(i,:));
            classes{s,t,im,2}=char(detections_.class2(i,:));
            classes{s,t,im,3}=char(detections_.class3(i,:));
            classes{s,t,im,4}=char(detections_.class4(i,:));
            classes{s,t,im,5}=char(detections_.class5(i,:));
            
            detections(s,t,im,1,1)=detections_.x1(i);
            detections(s,t,im,1,2)=detections_.y1(i);
            detections(s,t,im,1,3)=detections_.w1(i);
            detections(s,t,im,1,4)=detections_.h1(i);
            
            detections(s,t,im,2,1)=detections_.x2(i);
            detections(s,t,im,2,2)=detections_.y2(i);
            detections(s,t,im,2,3)=detections_.w2(i);
            detections(s,t,im,2,4)=detections_.h2(i);
            
            detections(s,t,im,3,1)=detections_.x3(i);
            detections(s,t,im,3,2)=detections_.y3(i);
            detections(s,t,im,3,3)=detections_.w3(i);
            detections(s,t,im,3,4)=detections_.h3(i);
            
            detections(s,t,im,4,1)=detections_.x4(i);
            detections(s,t,im,4,2)=detections_.y4(i);
            detections(s,t,im,4,3)=detections_.w4(i);
            detections(s,t,im,4,4)=detections_.h4(i);
            
            detections(s,t,im,5,1)=detections_.x5(i);
            detections(s,t,im,5,2)=detections_.y5(i);
            detections(s,t,im,5,3)=detections_.w5(i);
            detections(s,t,im,5,4)=detections_.h5(i);
        end
    end
end
