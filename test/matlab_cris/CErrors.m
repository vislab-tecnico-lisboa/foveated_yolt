function [C_sigmas,C_threshs,C_fix_pts,C_error5_pos,C_error5_av,C_error5_std,C_fb_error5_pos,C_fb_error5_av,C_fb_error5_std] = CErrors(images_number,top_k,...
    gt_classes,C_detections_file,C_fbdetections_file)

[C_sigmas,C_threshs, C_fix_pts, C_classes, ~, ~]=parse_detections(C_detections_file);
[~,~,~,C_fb_classes,~, ~]=parse_detections(C_fbdetections_file);
%[~,~,~,C_fb_classes,~, ~]=feedback_parse_detections(C_fbdetections_file);

[~, ~, C_error5_av, C_error5_std,~,C_error5_pos]=classification_error_rates(...
    C_sigmas,C_threshs,C_fix_pts,images_number,C_classes,gt_classes,top_k);
[~, ~, C_fb_error5_av,C_fb_error5_std,~,C_fb_error5_pos]=classification_error_rates(...
    C_sigmas,C_threshs,C_fix_pts,images_number,C_fb_classes,gt_classes,top_k);

% [~, ~, C_fb_error5_av, C_fb_error5_std,~,C_fb_error5_pos]=classification_error_rates(...
%     C_sigmas,C_threshs,C_fix_pts,images_number,C_fb_classes,gt_classes,top_k);

end

