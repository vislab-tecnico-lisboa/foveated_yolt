close all
%% Non Uniform Foveal Vision

f=60;
fov_min=3;
fov_max=101;
fovea = [fov_min:1:fov_max];
info = zeros(1,11);
N=5;
[x,y]=meshgrid(-113:113,-113:113);

A=zeros(N,1);
r=zeros(N,1);

% for i=0:N-1
%     fovi=exp(-0.5*(x.^2+y.^2)/f^(2*(i+1)));
%     A(i+1)=sum(sum(fovi))/(227^2);
%     r(i+1)=A(i+1)*(1/(2^(2*i)));
% end

for k=1:length(fovea)
    for i=0:N-1
        fovi=exp(-0.5*(x.^2+y.^2)/fovea(k)^(2*(i+1)));
        A(i+1)=sum(sum(fovi))/(227^2);
        r(i+1)=A(i+1)*(1/(2^(2*i)));
    end
    info(k)=sum(r);
end
%info=sum(r)
% figure(1)
% set(gcf, 'Color', [1,1,1]);
% fontsize=30;
% plot(fovea,info)
% xlabel('$\sigma_f$','Interpreter','LaTex','FontSize',fontsize);
% ylabel('Information Gain','Interpreter','LaTex','FontSize',fontsize);
% set(gca, 'FontSize', fontsize);

figure(3)
set(gcf, 'Color', [1,1,1]);
fontsize=30;
plot(fovea,1./info)
xlim([fov_min fov_max])

xlabel('$\sigma_f$','Interpreter','LaTex','FontSize',fontsize);
ylabel('Information Compression','Interpreter','LaTex','FontSize',fontsize);
set(gca, 'XTick',[fov_min 10:10:fov_max], 'YTick',0:5:100,'FontSize', fontsize);
saveas(figure(3),'fovea_info_reduction.png')


%% Uniform Vision

sigma = [0:0.5:10.01];
axis_y = [0:0.1:1];

reduced_info = 1./(sqrt(2*3.14)./(2*3.14*sigma));

figure(2)
set(gcf, 'Color', [1,1,1]);
fontsize=30;
plot(sigma,reduced_info)
xlabel('$\sigma_u$','Interpreter','LaTex','FontSize',fontsize);
ylabel('Information Compression','Interpreter','LaTex','FontSize',fontsize);
xlim([0 10])
%ylim([0 26])
set(gca, 'XTick',[0:1:10], 'YTick',[0:5:30], 'FontSize', fontsize);
saveas(figure(2),'uniform_info_reduction.png')



