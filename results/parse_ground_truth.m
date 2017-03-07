function gt = parse_ground_truth(gt_folder,images_number)

s = dir(fullfile(gt_folder, '*.xml'));
file_list = {s.name}';

%gt = repmat(struct(), length(file_list), 1 );
gt=zeros(images_number,4);
for i=1:images_number
    gt_ = xml2struct(strcat(gt_folder,file_list{i}));
    if length(gt_.annotation.object)>1
        for j=1:length(gt_.annotation.object)
            gt(i,1)=str2num(gt_.annotation.object{j}.bndbox.xmin.Text);
            gt(i,2)=str2num(gt_.annotation.object{j}.bndbox.ymin.Text);
            gt(i,3)=str2num(gt_.annotation.object{j}.bndbox.xmax.Text)-gt(i,1);
            gt(i,4)=str2num(gt_.annotation.object{j}.bndbox.ymax.Text)-gt(i,2);
        end
    else
            gt(i,1)=str2num(gt_.annotation.object.bndbox.xmin.Text);
            gt(i,2)=str2num(gt_.annotation.object.bndbox.ymin.Text);
            gt(i,3)=str2num(gt_.annotation.object.bndbox.xmax.Text)-gt(i,1);
            gt(i,4)=str2num(gt_.annotation.object.bndbox.ymax.Text)-gt(i,2);
    end
end