close all
%% Non Uniform Foveal Vision

f=60;
fov_min=0;
fov_max=101;
fovea = [fov_min:0.01:fov_max];
info = zeros(1,fov_max-fov_min);
N=5;
[x,y]=meshgrid(-113:113,-113:113);

A=zeros(N,1);
r=zeros(N,1);

for k=1:length(fovea)
    for i=0:N-1
        fovi=exp(-0.5*(x.^2+y.^2)/fovea(k)^(2*(i+1)));
        A(i+1)=sum(sum(fovi))/(227^2);
        r(i+1)=A(i+1)*(1/(2^(2*i)));
    end
    info(k)=sum(r);
end

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

sigma = [fov_min:1:fov_max];
axis_y = [0:0.1:1];

reduced_info = 1./(sqrt(2*3.14)./(2*3.14*sigma));

figure(2)
set(gcf, 'Color', [1,1,1]);
fontsize=30;
plot(sigma,reduced_info)
xlabel('$\sigma_u$','Interpreter','LaTex','FontSize',fontsize);
ylabel('Information Compression','Interpreter','LaTex','FontSize',fontsize);
xlim([0 fov_max])
%ylim([0 26])
set(gca, 'XTick',[0:1:fov_max], 'YTick',[0:5:100], 'FontSize', fontsize);
saveas(figure(2),'uniform_info_reduction.png')




%% Em decibel
figure(5)
set(gcf, 'Color', [1,1,1]);
fontsize=30;
semilogx(sigma,10*log(1./reduced_info), 'r-')
hold on
semilogx(fovea,10*log(info), 'b-')
xlabel('$\sigma$','Interpreter','LaTex','FontSize',fontsize);
ylabel('Gain [dB]','Interpreter','LaTex','FontSize',fontsize);
legend('Uniform Vision', 'Foveal Vision');
xlim([1 100]);
ylim([-100 0]);
%set(gca, 'XTick',[0:10:100], 'YTick',[-60:10:0], 'FontSize', fontsize);
set(gca, 'FontSize', fontsize);

saveas(figure(5), 'info_compress_db.png')
%export_fig -pdf info_compress_db
