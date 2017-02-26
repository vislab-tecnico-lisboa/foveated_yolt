#include <iostream>
#include <math.h>
#include <cmath>
#include <vector>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/contrib/contrib.hpp"
#include <string>

using namespace cv;
using namespace std;


class LaplacianBlending
{
public:
    cv::Mat image;
    int levels;
    
    std::vector<Mat> kernels;

    std::vector<Mat> imageLapPyr;
    std::vector<Mat> foveatedPyr;
    std::vector<cv::Mat> image_sizes;
    std::vector<cv::Mat> kernel_sizes;
    
    Mat imageSmallestLevel;
    Mat down,up;           
    Mat foveated_image;

    void buildPyramids()
    {

        imageLapPyr.clear();

        Mat currentImg = image;

        for (int l=0; l<levels; l++)
        {
            Mat image;
            pyrDown(currentImg, down);

            pyrUp(down, up, currentImg.size());

            Mat lap = currentImg - up;
            
            imageLapPyr[l]=lap;

            currentImg = down;
        }
        
        imageSmallestLevel=up;
        
    }


public:
    LaplacianBlending(const cv::Mat& _image,const int _levels, std::vector<Mat> _kernels):
        image(_image),levels(_levels), kernels(_kernels)
        {

            imageLapPyr.resize(levels);
            foveatedPyr.resize(levels);

            buildPyramids();

            image_sizes.resize(levels);
            kernel_sizes.resize(levels);

            for(int i=levels-1; i>=0;--i){

                cv::Mat image_size(2,1,CV_32S);
                
                image_size.at<int>(0,0)=imageLapPyr[i].cols;
                image_size.at<int>(1,0)=imageLapPyr[i].rows;
                image_sizes[i]=image_size;
                
                cv::Mat kernel_size(2,1,CV_32S);
                
                kernel_size.at<int>(0,0)=kernels[i].cols;
                kernel_size.at<int>(1,0)=kernels[i].rows;
                kernel_sizes[i]=kernel_size;
            }
        };


        
        // COMPUTE ROIS
        void computeRois(const cv::Mat & center, cv::Rect & kernel_roi_rect,
                         const cv::Mat & kernel_size, const cv::Mat & image_size)  {

            // Kernel center - image coordinate
            cv::Mat upper_left_kernel_corner = (kernel_size) / 2.0 - center;


            // encontrar roi no kernel
            // cv::Rect take (upper left corner, width, heigth)
            kernel_roi_rect=cv::Rect(upper_left_kernel_corner.at<int>(0,0),
                                     upper_left_kernel_corner.at<int>(1,0),
                                     image_size.at<int>(0,0),
                                     image_size.at<int>(1,0));
            }


        
        // FOVEATE
        cv::Mat foveate(const cv::Mat & center)
        {
            imageSmallestLevel.copyTo(foveated_image);
            cv::Rect kernel_roi_rect;
            

            for(int i=levels-1; i>=0; --i){

                cv::Rect image_roi_rect;
                cv::Rect kernel_roi_rect;           
                cv::Mat aux;

                if(i!=0){
                    aux=center/(powf(2,i));
                }
                else{
                    aux=center;
                }

                computeRois(aux,kernel_roi_rect,kernel_sizes[i],image_sizes[i]);

                // Multiplicar
                cv::Mat aux_pyr;
                imageLapPyr[i].copyTo(aux_pyr);
                cv::Mat result_roi;


                cv::multiply(aux_pyr,kernels[i](kernel_roi_rect),result_roi,1.0);

                result_roi.copyTo(aux_pyr);


                if(i==(levels-1)) {

                    add(foveated_image,aux_pyr,foveated_image);
                }

                else {

                    // pyrUp( tmp, dst, Size( tmp.cols*2, tmp.rows*2 ) )
                    pyrUp(foveated_image, foveated_image, Size(image_sizes[i].at<int>(0,0),image_sizes[i].at<int>(1,0)));
                    cv::add(foveated_image,aux_pyr,foveated_image);                   
                }
            }
            return foveated_image;
        }             
        
};

/*****************************************/
//                  FUNCTIONS
/*****************************************/


Mat createFilter(int m, int n, int sigma){

    Mat gkernel(m,n,CV_64FC3);

    double r,rx, ry, s = 2.0 * sigma * sigma;
    double xc= n*0.5;
    double yc= m*0.5;
    double max_value=-std::numeric_limits<double>::max();

    for (int x = 0; x < n; ++x) {
        rx = ((x-xc)*(x-xc));

        for(int y = 0; y < m; ++y) {

            ry = ((y-yc)*(y-yc));

            // FOR 3 CHANNELS
            gkernel.at<Vec3d>(y,x)[0] = exp(-(rx + ry)/s);
            gkernel.at<Vec3d>(y,x)[1] = exp(-(rx + ry)/s);
            gkernel.at<Vec3d>(y,x)[2] = exp(-(rx + ry)/s);

            if(gkernel.at<Vec3d>(y,x)[0]>max_value)
            {
                max_value=gkernel.at<Vec3d>(y,x)[0];
            }
        }
    }

    // normalize the Kernel
    for(int x = 0; x < n; ++x){
        for(int y = 0; y < m; ++y){

            // FOR 3 CHANNELS
            gkernel.at<Vec3d>(y,x)[0] /= max_value;
            gkernel.at<Vec3d>(y,x)[1] /= max_value;
            gkernel.at<Vec3d>(y,x)[2] /= max_value;

         }
    }

    return gkernel;
}


std::vector<Mat> createFilterPyr(int m, int n, int levels, int sigma){

    std::vector<Mat> kernels;
    Mat gkernel=createFilter(m,n,sigma);
    kernels.push_back(gkernel);

    for (int l=0; l<levels; ++l){

        Mat kernel_down;

        pyrDown(kernels[l], kernel_down);
        kernels.push_back(kernel_down);
    }
    return kernels;
}


