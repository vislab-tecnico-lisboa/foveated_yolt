%% Plot gaussian

mean=0;
sigma=1;
x=-3:0.01:3;
fx=1/sqrt(2*pi)/sigma*exp(-(x-mean).^2/2/sigma/sigma);

plot(x,fx)
ylabel('G(x)');
xlabel('x');
set(gcf, 'PaperPosition', [0 0 90 60]); %Position the plot further to the left and down. Extend the plot to fill entire paper.
set(gcf, 'PaperSize', [90 60]); %Keep the same paper size
saveas(gcf, 'gaussian_distribution', 'pdf')
%saveas(gcf, 'gaussian_sigma_1', 'pdf')


%% Plot original and blur image

figure();
I = imread('../files/ILSVRC2012_val_00000003.JPEG');
subplot(1,2,1);imshow(I);title('Original Image');
H = fspecial('Gaussian',[5 5],10);
GaussBlur = imfilter(I,H);
subplot(1,2,2);imshow(GaussBlur);title('Gaussian Blurred Image');



%% Plot gaussian filter
% 3D
figure();
sigma = 60;
G1=fspecial('gauss',[227 227], sigma);
[X,Y] = meshgrid(1:size(G1,2), 1:size(G1,1));
mesh(X, Y, G1);
ylim([0 227])
xlim([0 227])
xlabel('X'); ylabel('Y'); zlabel('Amplitude');
%title('3D visualization of the Gaussian filter');
colorbar;
set(gcf, 'PaperPosition', [0 0 90 60]); %Position the plot further to the left and down. Extend the plot to fill entire paper.
set(gcf, 'PaperSize', [90 60]); %Keep the same paper size
saveas(gcf, 'gaussian_filter_3d_sigma_60', 'pdf')


% 2D
figure(4);
sigma = 30;
G1=fspecial('gauss',[227 227], sigma);

image=imshow(G1,[]);
colorbar;


%saveas(image, 'gaussian_filter_sigma_30', 'pdf')




