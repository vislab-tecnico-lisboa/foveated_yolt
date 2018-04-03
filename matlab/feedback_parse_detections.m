function [feedback_sigmas,feedback_thres,rank_feedback_classes,feedback_classes,feedback_scores] = feedback_parse_detections(images_number,detections_file)
detections_ = tdfread(detections_file,';');

feedback_sigmas=unique(detections_.sigma);
feedback_thres=unique(detections_.thres);

feedback_scores=zeros(length(detections_.score1),25);
feedback_scores(:,1)=detections_.score1;
feedback_scores(:,2)=detections_.score2;
feedback_scores(:,3)=detections_.score3;
feedback_scores(:,4)=detections_.score4;
feedback_scores(:,5)=detections_.score5;
feedback_scores(:,6)=detections_.score6;
feedback_scores(:,7)=detections_.score7;
feedback_scores(:,8)=detections_.score8;
feedback_scores(:,9)=detections_.score9;
feedback_scores(:,10)=detections_.score10;
feedback_scores(:,11)=detections_.score11;
feedback_scores(:,12)=detections_.score12;
feedback_scores(:,13)=detections_.score13;
feedback_scores(:,14)=detections_.score14;
feedback_scores(:,15)=detections_.score15;
feedback_scores(:,16)=detections_.score16;
feedback_scores(:,17)=detections_.score17;
feedback_scores(:,18)=detections_.score18;
feedback_scores(:,19)=detections_.score19;
feedback_scores(:,20)=detections_.score20;
feedback_scores(:,21)=detections_.score21;
feedback_scores(:,22)=detections_.score22;
feedback_scores(:,23)=detections_.score23;
feedback_scores(:,24)=detections_.score24;
feedback_scores(:,25)=detections_.score25;

%feedback_detections=zeros(length(feedback_sigmas),length(feedback_thres),images_number,5,4);   % feedback bbox
feedback_classes=cell(length(feedback_sigmas),length(feedback_thres),images_number,25);        % top 25 classes

total_images=length(detections_.sigma)/(length(feedback_sigmas)*length(feedback_thres));

for s=1:length(feedback_sigmas)
    for t=1:length(feedback_thres)
        for im=1:images_number
            i=im+(t-1)*total_images +(s-1)*total_images*length(feedback_thres);
            feedback_classes{s,t,im,1}=char(detections_.class1(i,:));
            feedback_classes{s,t,im,2}=char(detections_.class2(i,:));
            feedback_classes{s,t,im,3}=char(detections_.class3(i,:));
            feedback_classes{s,t,im,4}=char(detections_.class4(i,:));
            feedback_classes{s,t,im,5}=char(detections_.class5(i,:));
            feedback_classes{s,t,im,6}=char(detections_.class6(i,:));
            feedback_classes{s,t,im,7}=char(detections_.class7(i,:));
            feedback_classes{s,t,im,8}=char(detections_.class8(i,:));
            feedback_classes{s,t,im,9}=char(detections_.class9(i,:));
            feedback_classes{s,t,im,10}=char(detections_.class10(i,:));
            feedback_classes{s,t,im,11}=char(detections_.class11(i,:));
            feedback_classes{s,t,im,12}=char(detections_.class12(i,:));
            feedback_classes{s,t,im,13}=char(detections_.class13(i,:));
            feedback_classes{s,t,im,14}=char(detections_.class14(i,:));
            feedback_classes{s,t,im,15}=char(detections_.class15(i,:));
            feedback_classes{s,t,im,16}=char(detections_.class16(i,:));
            feedback_classes{s,t,im,17}=char(detections_.class17(i,:));
            feedback_classes{s,t,im,18}=char(detections_.class18(i,:));
            feedback_classes{s,t,im,19}=char(detections_.class19(i,:));
            feedback_classes{s,t,im,20}=char(detections_.class20(i,:));
            feedback_classes{s,t,im,21}=char(detections_.class21(i,:));
            feedback_classes{s,t,im,22}=char(detections_.class22(i,:));
            feedback_classes{s,t,im,23}=char(detections_.class23(i,:));
            feedback_classes{s,t,im,24}=char(detections_.class24(i,:));
            feedback_classes{s,t,im,25}=char(detections_.class25(i,:));
            
%             feedback_detections(s,t,im,1,1)=detections_.x1(i);
%             feedback_detections(s,t,im,1,2)=detections_.y1(i);
%             feedback_detections(s,t,im,1,3)=detections_.w1(i);
%             feedback_detections(s,t,im,1,4)=detections_.h1(i);
%             
%             feedback_detections(s,t,im,2,1)=detections_.x2(i);
%             feedback_detections(s,t,im,2,2)=detections_.y2(i);
%             feedback_detections(s,t,im,2,3)=detections_.w2(i);
%             feedback_detections(s,t,im,2,4)=detections_.h2(i);
%             
%             feedback_detections(s,t,im,3,1)=detections_.x3(i);
%             feedback_detections(s,t,im,3,2)=detections_.y3(i);
%             feedback_detections(s,t,im,3,3)=detections_.w3(i);
%             feedback_detections(s,t,im,3,4)=detections_.h3(i);
%             
%             feedback_detections(s,t,im,4,1)=detections_.x4(i);
%             feedback_detections(s,t,im,4,2)=detections_.y4(i);
%             feedback_detections(s,t,im,4,3)=detections_.w4(i);
%             feedback_detections(s,t,im,4,4)=detections_.h4(i);
%             
%             feedback_detections(s,t,im,5,1)=detections_.x5(i);
%             feedback_detections(s,t,im,5,2)=detections_.y5(i);
%             feedback_detections(s,t,im,5,3)=detections_.w5(i);
%             feedback_detections(s,t,im,5,4)=detections_.h5(i);
                
        end
    end
end
% 
% 
rank_feedback_classes = cell(length(feedback_sigmas),length(feedback_thres),images_number,5);

for s=1:length(feedback_sigmas)
    for t=1:length(feedback_thres)
        for im=1:images_number

            % Rank 25 predicted class labels to top5 final solution
            [rank_feedback_scores, rank_score_index] = sort(feedback_scores(im,:), 'descend');
            
            aux = unique(char(feedback_classes{s,t,im,rank_score_index(:)}),'rows','stable');
            rank_feedback_classes{s,t,im,1}=aux(1,:);
            rank_feedback_classes{s,t,im,2}=aux(2,:);
            rank_feedback_classes{s,t,im,3}=aux(3,:);
            rank_feedback_classes{s,t,im,4}=aux(4,:);
            rank_feedback_classes{s,t,im,5}=aux(5,:);

%             
%             feedback_detections(s,t,im,1,1)=detections_.x1(i);
%             feedback_detections(s,t,im,1,2)=detections_.y1(i);
%             feedback_detections(s,t,im,1,3)=detections_.w1(i);
%             feedback_detections(s,t,im,1,4)=detections_.h1(i);
%             
%             feedback_detections(s,t,im,2,1)=detections_.x2(i);
%             feedback_detections(s,t,im,2,2)=detections_.y2(i);
%             feedback_detections(s,t,im,2,3)=detections_.w2(i);
%             feedback_detections(s,t,im,2,4)=detections_.h2(i);
%             
%             feedback_detections(s,t,im,3,1)=detections_.x3(i);
%             feedback_detections(s,t,im,3,2)=detections_.y3(i);
%             feedback_detections(s,t,im,3,3)=detections_.w3(i);
%             feedback_detections(s,t,im,3,4)=detections_.h3(i);
%             
%             feedback_detections(s,t,im,4,1)=detections_.x4(i);
%             feedback_detections(s,t,im,4,2)=detections_.y4(i);
%             feedback_detections(s,t,im,4,3)=detections_.w4(i);
%             feedback_detections(s,t,im,4,4)=detections_.h4(i);
%             
%             feedback_detections(s,t,im,5,1)=detections_.x5(i);
%             feedback_detections(s,t,im,5,2)=detections_.y5(i);
%             feedback_detections(s,t,im,5,3)=detections_.w5(i);
%             feedback_detections(s,t,im,5,4)=detections_.h5(i);     
%             
%             
%             
         end
     end
end