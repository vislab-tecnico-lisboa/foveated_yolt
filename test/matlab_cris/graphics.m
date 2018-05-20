%% Graphics for Classification Errors

%load ('class_errors.mat');

% Classification Vs Position
data = [C_error_pos(:,1:8);C_error_pos(:,9:16);C_error_pos(:,17:24);C_error_pos(:,25:32);...
        C_error_pos(:,33:40);C_error_pos(:,41:48);C_error_pos(:,49:56);C_error_pos(:,57:64)];
data2 = [C_fb_error_pos(:,1:8);C_fb_error_pos(:,9:16);C_fb_error_pos(:,17:24);C_fb_error_pos(:,25:32);...
        C_fb_error_pos(:,33:40);C_fb_error_pos(:,41:48);C_fb_error_pos(:,49:56);C_fb_error_pos(:,57:64)];
figure(1)
heatmap(C_fix_pts(1:8),C_fix_pts(1:8),data*100,'Colormap',bone,'XLabel',...
    'u','FontSize', 24,'YLabel','v','Title','Classification Error(%)',...
    'ColorLimits',[35 57],'CellLabelColor','none','Grid','off');
figure(2)
heatmap(C_fix_pts(1:8),C_fix_pts(1:8),data2*100,'Colormap',bone,'XLabel',...
    'u','FontSize', 24,'YLabel','v','Title','Classification Error(%)',...
    'ColorLimits',[35 57],'CellLabelColor','none','Grid','off');
%%
% Classification Vs Sigmas
figure(3)
plot(C_sigmas0,100*C_error5_av0, 'b-');
hold on
plot(C_sigmas0,100*C_fb_error5_av0,'r-');
hold on
errorbar(C_sigmas,100*C_error5_av,100*C_error5_std, 'g-');
hold on
errorbar(C_sigmas,100*C_fb_error5_av,100*C_fb_error5_std,'m-');
hold on
plot(C_sigmas0,100*C_baseline*ones(size(C_sigmas0)),'k--');
legend({'1st pass-central point','2nd pass-central point','1st pass-16 points','2nd pass-16 points','Baseline'},'FontSize', 20);
set(gca, 'FontSize', 20);
grid on;
xlabel('f_0','FontSize', 24);
ylabel('Classification Error (%)','FontSize', 24);
ylim([0 100]);

%% Graphics for Localization Errors

load ('loc_errors_new.mat');

% Localization Vs Position

data = [L_error_pos(:,1:8);L_error_pos(:,9:16);L_error_pos(:,17:24);L_error_pos(:,25:32);...
        L_error_pos(:,33:40);L_error_pos(:,41:48);L_error_pos(:,49:56);L_error_pos(:,57:64)];
    
data2 = [L_fb_error_pos(:,1:8);L_fb_error_pos(:,9:16);L_fb_error_pos(:,17:24);L_fb_error_pos(:,25:32);...
        L_fb_error_pos(:,33:40);L_fb_error_pos(:,41:48);L_fb_error_pos(:,49:56);L_fb_error_pos(:,57:64)];

figure(1)
heatmap(L_fix_pts(1:8),L_fix_pts(1:8),data*100,'Colormap',bone,'XLabel',...
    'x','FontSize', 24,'YLabel','y','Title','Localization Error(%) in the 1st pass',...
    'ColorLimits',[58 65]);
figure(2)
heatmap(L_fix_pts(1:8),L_fix_pts(1:8),data2*100,'Colormap',bone,'XLabel',...
    'x','FontSize', 24,'YLabel','y','Title','Localization Error(%) in the 2nd pass',...
    'ColorLimits',[58 65]);
%%
figure(3)
plot(L_threshs0,100*L_error_av0, 'b-');
hold on
plot(L_threshs0,100*L_fb_error_av0,'r-');
hold on
errorbar(L_threshs,100*L_error_av,100*L_error_std, 'g-');
hold on
errorbar(L_threshs,100*L_fb_error_av,100*L_fb_error_std,'m-');
set(gca, 'FontSize', 20);
grid on;
legend({'1st pass-central point','2nd pass-central point','1st pass-16 points','2nd pass-16 points'},'FontSize', 24);
xlabel('\theta','FontSize', 24);
ylabel('Localization Error (%)','FontSize', 24);
ylim([0 100]);

%%   
    
figure(4)
plot(L_sigmas0,100*L_error_av0_s, 'b-');
hold on
plot(L_sigmas0,100*L_fb_error_av0_s,'r-');
hold on
errorbar(L_sigmas,100*L_error_av_s,100*L_error_std_s, 'g-');
hold on
errorbar(L_sigmas,100*L_fb_error_av_s,100*L_fb_error_std_s,'m-');
hold on
plot(L_sigmas,100*L_baseline*ones(size(L_sigmas)),'k--');
set(gca, 'FontSize', 20);
grid on;
legend({'1st pass-central point','2nd pass-central point','1st pass-16 points','2nd pass-16 points','Baseline'},'FontSize', 20);
xlabel('f_0','FontSize', 24);
ylabel('Localization Error (%)','FontSize', 24);
ylim([0 100]);

%%
figure(5)
plot(C_threshs0,100*C_error5_av0_t, 'b-');
hold on
plot(C_threshs0,100*C_fb_error5_av0_t,'r-');
hold on
errorbar(C_threshs,100*C_error5_av_t,100*C_error5_std_t, 'g-');
hold on
errorbar(C_threshs,100*C_fb_error5_av_t,100*C_fb_error5_std_t,'m-');
hold on
plot(C_threshs,100*C_baseline*ones(size(C_threshs)),'k--');
legend({'1st pass-central point','2nd pass-central point','1st pass-16 points','2nd pass-16 points','Baseline'},'FontSize', 20);
xlabel('th','FontSize', 24);
ylabel('Classification Error (%)','FontSize', 24);
ylim([0 100]);
