function [sigmas,thres,classes,scores] = parse_detections2(images_number,detections_file)
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
            
           
        end
    end
end
