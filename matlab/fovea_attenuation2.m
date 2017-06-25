close all
hold on
%% Uniform Vision
% P = 1/(4.pi.sigma²)
sigmas_0 = [1.0:0.5:110.1];

% mag2db
P = 1./(sqrt(4*3.14*sigmas_0.^2));


% CARTESIAN
figure(1)
set(gcf, 'Color', [1,1,1]);
fontsize=30;
semilogx(sigmas_0,10*log(P))
xlabel('$\sigma_u$','Interpreter','LaTex','FontSize',fontsize);
ylabel('Information Gain','Interpreter','LaTex','FontSize',fontsize);
%xlim([0 10])
%ylim([0 26])
%set(gca, 'XTick',[0:1:100], 'YTick',[0:0.2:1], 'FontSize', fontsize);
saveas(figure(1),'uniform_info_reduction.png')


%% Foveal
% Filtering - 1/(4*pi*sigma²) -- sigma, P

% spatial weighting

f=60;
fov_min=1;
fov_max=101;
fovea = [fov_min:1:fov_max];
info = zeros(1,101);
levels=5;
[x,y]=meshgrid(-113:113,-113:113);

%r=zeros(level,1);
T=zeros(length(sigmas_0),1);

for i=1:length(sigmas_0)
    T_=0;
    for k=0:levels-1
        sigma_k=2^k * sigmas_0(i);
        f_k=2^k * sigmas_0(i);
        fovi=exp(-0.5*(x.^2+y.^2)/sigma_k^2);
        R=sum(sum(fovi))/(227^2); 
        P=1/(4*pi*sigma_k^2);
        T_=T_+R;%+P;
        %r(i+1)=A(i+1)*(1/(2^(2*i)));
    end
    T(i)=T_;

    %info(k)=sum(r);
end


%eq. 4.18

%Foveal
figure(2)
set(gcf, 'Color', [1,1,1]);
%fontsize=30;
semilogx(sigmas_0,10*log(T))
xlabel('$\sigma_f$','Interpreter','LaTex','FontSize',fontsize);
ylabel('Information Gain','Interpreter','LaTex','FontSize',fontsize);
set(gca, 'FontSize', fontsize)



