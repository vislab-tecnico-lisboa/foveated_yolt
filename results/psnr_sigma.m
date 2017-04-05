
ref = imread('../Figures/puma_original.JPEG');


A = imread('../Figures/psnr_images/puma_blur_1.JPEG');
B = imread('../Figures/psnr_images/puma_blur_2.JPEG');
C = imread('../Figures/psnr_images/puma_blur_3.JPEG');
D = imread('../Figures/psnr_images/puma_blur_4.JPEG');
E = imread('../Figures/psnr_images/puma_blur_5.JPEG');
F = imread('../Figures/psnr_images/puma_blur_6.JPEG');
G = imread('../Figures/psnr_images/puma_blur_7.JPEG');
H = imread('../Figures/psnr_images/puma_blur_8.JPEG');
I = imread('../Figures/psnr_images/puma_blur_9.JPEG');
J = imread('../Figures/psnr_images/puma_blur_10.JPEG');

%% Peak Signal-to-Noise Ratio 

%[peaksnr, snr] = psnr(A, ref)

sigmas = [1 2 3 4 5 6 7 8 9 10];
array_psnr = [21.1969  20.0054  19.5902 19.2689  18.9759 18.6994 18.4334 18.1864 17.9579 17.7454];
array_snr = [14.1472  12.9557 12.5405 12.2192 11.9263 11.6497 11.3837 11.1368 10.9083 10.6957];

figure(1)
fontsize=20;
set(gcf, 'Color', [1,1,1]);
plot(sigmas,array_psnr,'m--o'); 
xlabel('$\sigma$','Interpreter','LaTex','FontSize',fontsize);
ylabel('PSNR [dB]','Interpreter','LaTex','FontSize',fontsize);
xlim([1 10])
saveas(figure(1),'puma_psnr_sigma.png')

figure(2)
fontsize=20;
set(gcf, 'Color', [1,1,1]);
plot(sigmas,array_snr,'b--o'); 
xlabel('$\sigma$','Interpreter','LaTex','FontSize',fontsize);
ylabel('SNR [dB]','Interpreter','LaTex','FontSize',fontsize);
xlim([1 10])
saveas(figure(2),'puma_snr_sigma.png')



%% Compression Ratio
% ratio between the uncompressed size and compressed size
orig = imfinfo('../Figures/puma_original.JPEG');
temp1 = imfinfo('../Figures/psnr_images/puma_blur_1.JPEG');
temp2 = imfinfo('../Figures/psnr_images/puma_blur_2.JPEG');
temp3 = imfinfo('../Figures/psnr_images/puma_blur_3.JPEG');
temp4 = imfinfo('../Figures/psnr_images/puma_blur_4.JPEG');
temp5 = imfinfo('../Figures/psnr_images/puma_blur_5.JPEG');
temp6 = imfinfo('../Figures/psnr_images/puma_blur_6.JPEG');
temp7 = imfinfo('../Figures/psnr_images/puma_blur_7.JPEG');
temp8 = imfinfo('../Figures/psnr_images/puma_blur_8.JPEG');
temp9 = imfinfo('../Figures/psnr_images/puma_blur_9.JPEG');
temp10 = imfinfo('../Figures/psnr_images/puma_blur_10.JPEG');
%original = (orig.Width * orig.Height * orig.BitDepth)/8
%compress = (temp.Width * temp.Height * temp.BitDepth)/8
compression_ratio = [orig.FileSize/temp1.FileSize orig.FileSize/temp2.FileSize orig.FileSize/temp3.FileSize orig.FileSize/temp4.FileSize orig.FileSize/temp5.FileSize ...
                    orig.FileSize/temp6.FileSize orig.FileSize/temp7.FileSize orig.FileSize/temp8.FileSize orig.FileSize/temp9.FileSize orig.FileSize/temp10.FileSize]


figure(3)
fontsize=20;
set(gcf, 'Color', [1,1,1]);
plot(sigmas,compression_ratio,'b--o'); 
xlabel('$\sigma$','Interpreter','LaTex','FontSize',fontsize);
ylabel('Compression Ratio','Interpreter','LaTex','FontSize',fontsize);
xlim([1 10])
saveas(figure(3),'puma_cr_sigma.png')

