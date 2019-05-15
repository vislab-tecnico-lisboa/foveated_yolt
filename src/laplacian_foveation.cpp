#include "laplacian_foveation.hpp"

LaplacianBlending::LaplacianBlending(const int & _width, const int & _height, const int & _levels, const int & _sigma_x, const int & _sigma_y, const int & _sigma_xy) : width(_width), height(_height), levels(_levels)
{
	image_lap_pyr.resize(levels);
	foveated_pyr.resize(levels);
	image_sizes.resize(levels);
	kernel_sizes.resize(levels);
	kernels.resize(levels+1);

	CreateFilterPyr(width, height, _sigma_x, _sigma_y, _sigma_xy);

}

LaplacianBlending::~LaplacianBlending() {
	std::vector<cv::Mat>().swap(image_lap_pyr);
	std::vector<cv::Mat>().swap(foveated_pyr);
	std::vector<cv::Mat>().swap(kernels);
	std::vector<cv::Mat>().swap(image_sizes);
	std::vector<cv::Mat>().swap(kernel_sizes);
}

void LaplacianBlending::ComputeRois(const cv::Mat &center, cv::Rect &kernel_roi_rect, const cv::Mat &kernel_size, const cv::Mat &image_size) {
    // Kernel center - image coordinate
    cv::Mat upper_left_kernel_corner = ((kernel_size) / 2.0) - center;

    // encontrar roi no kernel
    // cv::Rect take (upper left corner, width, heigth)
    kernel_roi_rect=cv::Rect(upper_left_kernel_corner.at<int>(0,0), upper_left_kernel_corner.at<int>(1,0), image_size.at<int>(0,0), image_size.at<int>(1,0));
}

void LaplacianBlending::BuildPyramids(const cv::Mat & image) {

    cv::Mat current_img=image;
    current_img.convertTo(current_img, CV_64FC3); 

    for (int l=0; l<levels; ++l) {
        cv::pyrDown(current_img, down);
        cv::pyrUp(down, up, current_img.size());
        image_lap_pyr[l]=current_img-up;
        current_img = down;
    }
            
    image_smallest_level=up;           
}

cv::Mat LaplacianBlending::Foveate(const cv::Mat &image, const cv::Mat &center) {

    BuildPyramids(image);
    image_smallest_level.copyTo(foveated_image);

    for(int i=levels-1; i>=0; --i) {
        cv::Mat aux;
        aux=center/(powf(2,i));
        cv::Rect kernel_roi_rect;         

        cv::Mat result_roi;
        cv::Mat aux_pyr;
        
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



cv::Mat LaplacianBlending::CreateFilter(const int & m, const int & n, const int & sigma_x, const int & sigma_y, const int & sigma_xy) {

    cv::Mat gkernel(m,n,CV_64FC3);

    double rho=(double)sigma_xy/(sigma_x*sigma_y);

    double term=-1.0/(2.0*(1.0-(rho*rho)));
    double s_x = term/(sigma_x*sigma_x);
    double s_y = term/(sigma_y*sigma_y);
    double s_xy = 2.0*rho*term/(sigma_x*sigma_y);

    double xc = n*0.5;
    double yc = m*0.5;
    double max_value = -std::numeric_limits<double>::max();

    // build kernel

    for (unsigned int x=0; x<n; ++x) 
    {
        double dx=(x-xc);
        double rx = dx*dx;
        
        for(unsigned int y=0; y<m; ++y) 
	{
	    double dy=(y-yc);
            double ry=dy*dy;
            double rxy=dx*dy;
	    double expression=exp((rx*s_x) + (ry*s_y) - (rxy*s_xy));

            // FOR 3 CHANNELS
            gkernel.at<Vec3d>(y,x)[0] = expression;
            gkernel.at<Vec3d>(y,x)[1] = expression;
            gkernel.at<Vec3d>(y,x)[2] = expression;

            if(expression>max_value)
                max_value=expression;
        }
    }

    // normalize the Kernel
    for(unsigned int x=0; x<n; ++x) {
        for(unsigned int y=0; y<m; ++y) {

            // FOR 3 CHANNELS
            gkernel.at<Vec3d>(y,x)[0] /= max_value;
            gkernel.at<Vec3d>(y,x)[1] /= max_value;
            gkernel.at<Vec3d>(y,x)[2] /= max_value;
        }
    }

    return gkernel;
}


void LaplacianBlending::CreateFilterPyr(const int & width, const int & height, const int & _sigma_x, const int & _sigma_y, const int & sigma_xy) 
{
	// Foveate images
	int m=floor(4*height);
	int n=floor(4*width);

	kernels[0]=CreateFilter(m,n,_sigma_x,_sigma_y,sigma_xy);

	for (int l=0; l<levels; ++l) 
	{
		cv::pyrDown(kernels[l], kernels[l+1]);

		cv::Mat image_size(2,1,CV_32S);
		image_size.at<int>(0,0)=ceil(width/powf(2.0,l));
		image_size.at<int>(1,0)=ceil(height/powf(2.0,l));
		image_sizes[l]=image_size;

		cv::Mat kernel_size(2,1,CV_32S);    
		kernel_size.at<int>(0,0)=kernels[l].cols;
		kernel_size.at<int>(1,0)=kernels[l].rows;
		kernel_sizes[l]=kernel_size;
	}
}


