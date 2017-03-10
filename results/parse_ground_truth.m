function gt = parse_ground_truth(gt_folder,images_number)

s = dir(fullfile(gt_folder, '*.xml'));
file_list = {s.name}';

gt=[];
for i=1:images_number
    gt_ = xml2struct(strcat(gt_folder,file_list{i}));
    size=zeros(1,2);
    size(1)=str2num(gt_.annotation.size.width.Text);
    size(2)=str2num(gt_.annotation.size.height.Text);
    gt(i).size=size;
    gt(i).filename=strcat(gt_.annotation.filename.Text,'.JPEG');
    if length(gt_.annotation.object)>1
        gt(i).bboxes=zeros(length(gt_.annotation.object),4);
        for j=1:length(gt_.annotation.object)
            bbox=zeros(1,4);
            bbox(1)=str2num(gt_.annotation.object{j}.bndbox.xmin.Text);
            bbox(2)=str2num(gt_.annotation.object{j}.bndbox.ymin.Text);
            bbox(3)=str2num(gt_.annotation.object{j}.bndbox.xmax.Text)-bbox(1);
            bbox(4)=str2num(gt_.annotation.object{j}.bndbox.ymax.Text)-bbox(2);
            gt(i).bboxes(j,:)=bbox;
        end
    else
        gt(i).bboxes=zeros(1,4);
        
        bbox=zeros(1,4);
        bbox(1)=str2num(gt_.annotation.object.bndbox.xmin.Text);
        bbox(2)=str2num(gt_.annotation.object.bndbox.ymin.Text);
        bbox(3)=str2num(gt_.annotation.object.bndbox.xmax.Text)-bbox(1);
        bbox(4)=str2num(gt_.annotation.object.bndbox.ymax.Text)-bbox(2);
        gt(i).bboxes=bbox;
    end
end