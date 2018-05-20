function [L_threshs,L_sigmas, L_fix_pts,L_error_pos,L_error_av,L_error_std,L_fb_error_pos,L_fb_error_av,L_fb_error_std] = LErrors(...
    images_number,detections_resolution,overlap_correct,top_k,gt_detections,L_detections_file,L_fbdetections_file)

[L_sigmas,L_threshs, L_fix_pts, L_classes, L_scores, L_detections]=parse_detections(L_detections_file);
[~,~, ~, L_fb_classes, L_fb_scores, L_fb_detections]=parse_detections(L_fbdetections_file);

[L_error_pos,L_error_av,L_error_std] = detection_error_rates(L_sigmas,L_threshs,L_fix_pts,images_number,L_detections,...
    gt_detections,detections_resolution,top_k,overlap_correct);
[L_fb_error_pos,L_fb_error_av,L_fb_error_std] = detection_error_rates(L_sigmas,L_threshs,L_fix_pts,images_number,L_fb_detections,...
    gt_detections,detections_resolution,top_k,overlap_correct);

end

