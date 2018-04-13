
#include "laplacian_foveation.hpp"

LaplacianBlending::LaplacianBlending(const cv::Mat &_image, const int _levels, const int _sigma) {
    
    image=_image;
    levels=_levels; 

    // Foveate images
    int m=floor(4*_image.size().height);
    int n=floor(4*_image.size().width);

    CreateFilterPyr(m, n, _levels, _sigma);

    image_lap_pyr.resize(levels);
    foveated_pyr.resize(levels);
    image_sizes.resize(levels);
    kernel_sizes.resize(levels);
    
    BuildPyramids();
    
    for(int i=levels-1; i>=0; --i) {  
        
        cv::Mat image_size(2,1,CV_32S);
	    image_size.at<int>(0,0)=image_lap_pyr[i].cols;
	    image_size.at<int>(1,0)=image_lap_pyr[i].rows;
	    image_sizes[i]=image_size;
	    
        cv::Mat kernel_size(2,1,CV_32S);    
        kernel_size.at<int>(0,0)=kernels[i].cols;
	    kernel_size.at<int>(1,0)=kernels[i].rows;
	    kernel_sizes[i]=kernel_size;
    }
}

LaplacianBlending::~LaplacianBlending() {
	std::vector<cv::Mat>().swap(image_lap_pyr);
    std::vector<cv::Mat>().swap(foveated_pyr);
    std::vector<cv::Mat>().swap(kernels);
    std::vector<cv::Mat>().swap(image_sizes);
    std::vector<cv::Mat>().swap(kernel_sizes);
}

void LaplacianBlending::BuildPyramids() {

    cv::Mat current_img=image;
    cv::Mat lap;

    for (int l=0; l<levels; l++) {

        cv::pyrDown(current_img, down);
        cv::pyrUp(down, up, current_img.size());
        lap=current_img-up;
        
        image_lap_pyr[l]=lap.clone();
        current_img = down;
    }
            
    image_smallest_level=up;
  
}

void LaplacianBlending::ComputeRois(const cv::Mat &center, cv::Rect &kernel_roi_rect,
                                    const cv::Mat &kernel_size, const cv::Mat &image_size) {

    // Kernel center - image coordinate
    cv::Mat upper_left_kernel_corner = (kernel_size) / 2.0 - center;

    // encontrar roi no kernel
    // cv::Rect take (upper left corner, width, heigth)
    kernel_roi_rect=cv::Rect(upper_left_kernel_corner.at<int>(0,0),
                             upper_left_kernel_corner.at<int>(1,0),
                             image_size.at<int>(0,0),
                             image_size.at<int>(1,0));
}


cv::Mat LaplacianBlending::Foveate(const cv::Mat &center) {
    
    image_smallest_level.copyTo(foveated_image);

    for(int i=levels-1; i>=0; --i) {
        
        cv::Rect image_roi_rect;  
        cv::Rect kernel_roi_rect;         
        cv::Mat aux;

        cv::Mat result_roi;
        cv::Mat aux_pyr;
        
        if(i!=0)
            aux=center/(powf(2,i));
        else
            aux=center;

        ComputeRois(aux, kernel_roi_rect, kernel_sizes[i], image_sizes[i]);
            
        // Multiplicar
        image_lap_pyr[i].copyTo(aux_pyr);
        cv::multiply(aux_pyr, kernels[i](kernel_roi_rect), result_roi, 1.0, aux_pyr.type());
        result_roi.copyTo(aux_pyr);

        if(i==(levels-1))
            cv::add(foveated_image,aux_pyr,foveated_image);
        else {
            cv::pyrUp(foveated_image, foveated_image, Size(image_sizes[i].at<int>(0,0),image_sizes[i].at<int>(1,0)));
            cv::add(foveated_image,aux_pyr,foveated_image);                   
        }
    }
        
    return foveated_image;
}

cv::Mat LaplacianBlending::CreateFilter(int m, int n, int sigma) {

    cv::Mat gkernel(m,n,CV_64FC3);

    double r, rx, ry;
    double s = 2.0*sigma*sigma;
    double xc = n*0.5;
    double yc = m*0.5;
    double max_value = -std::numeric_limits<double>::max();

    for (int x=0; x<n; ++x) {
        
        rx = ((x-xc)*(x-xc));
        
        for(int y=0; y<m; ++y) {

            ry=((y-yc)*(y-yc));

            // FOR 3 CHANNELS
            gkernel.at<Vec3d>(y,x)[0] = exp(-(rx + ry)/s);
            gkernel.at<Vec3d>(y,x)[1] = exp(-(rx + ry)/s);
            gkernel.at<Vec3d>(y,x)[2] = exp(-(rx + ry)/s);

            if(gkernel.at<Vec3d>(y,x)[0]>max_value)
                max_value=gkernel.at<Vec3d>(y,x)[0];
        }
    }

    // normalize the Kernel
    for(int x=0; x<n; ++x) {
        for(int y=0; y<m; ++y) {

            // FOR 3 CHANNELS
            gkernel.at<Vec3d>(y,x)[0] /= max_value;
            gkernel.at<Vec3d>(y,x)[1] /= max_value;
            gkernel.at<Vec3d>(y,x)[2] /= max_value;
        }
    }

    return gkernel;
}

void LaplacianBlending::CreateFilterPyr(int m, int n, int levels, int sigma) {

    cv::Mat gkernel=CreateFilter(m,n,sigma);
    kernels.push_back(gkernel);

    for (int l=0; l<levels; ++l) {
        
        cv::Mat kernel_down;
        cv::pyrDown(kernels[l], kernel_down);
        kernels.push_back(kernel_down);
    }
}