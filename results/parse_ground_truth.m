function gt = parse_ground_truth(gt_folder,images_number)

s = dir(fullfile(gt_folder, '*.xml'));
file_list = {s.name}';

%gt = repmat(struct(), length(file_list), 1 );
gt=[];
for i=1:images_number
  gt = [gt; xml2struct(strcat(gt_folder,file_list{i}))];  
end