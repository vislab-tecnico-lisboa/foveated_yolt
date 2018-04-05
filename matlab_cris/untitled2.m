clear all
load('teste.mat');

%% Classification Error
% 1st pass
[~, ~, C_error5_av, ~]=classification_error_rates2(...
    C_sigmas,C_threshs,C_fix_pts,images_number,C_classes,gt_classes,top_k);
% 2nd pass Top 5 - Ranked with != labels
[~, ~, C_fb_error5_r_av,~]=classification_error_rates2(...
    C_sigmas,C_threshs,C_fix_pts,images_number,C_fb_rankclasses,gt_classes,top_k);
%%
figure(5)
data = [C_error5_av(:,1:4);C_error5_av(:,5:8);C_error5_av(:,9:12);C_error5_av(:,13:16)]
heatmap(C_fix_pts(1:4),C_fix_pts(1:4),data,'Colormap',bone,'XLabel',...
    'x','YLabel','y','Title','Classification Error with Position - 1st pass',...
    'ColorLimits',[0.5 0.7]);

figure(6)
data = [C_fb_error5_r_av(:,1:4);C_fb_error5_r_av(:,5:8);C_fb_error5_r_av(:,9:12);C_fb_error5_r_av(:,13:16)]
heatmap(C_fix_pts(1:4),C_fix_pts(1:4),data,'Colormap',bone,'XLabel',...
    'x','YLabel','y','Title','Classification Error with Position - 2nd pass',...
    'ColorLimits',[0.5 0.7]);

