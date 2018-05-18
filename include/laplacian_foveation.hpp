#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

class LaplacianBlending {

    public:
        LaplacianBlending(const cv::Mat &_image, const int _levels, const int _sigma_x, const int _sigma_y);
        ~LaplacianBlending();

        void BuildPyramids();
        void ComputeRois(const cv::Mat &center, cv::Rect &kernel_roi_rect,
                         const cv::Mat &kernel_size, const cv::Mat &image_size);
        cv::Mat Foveate(const cv::Mat &center);
        cv::Mat CreateFilter(int m, int n, int sigma_x, int sigma_y);
        void CreateFilterPyr(int m, int n, int levels, int sigma_x, int sigma_y);

    private:
        cv::Mat image;
        int levels;
        
        std::vector<cv::Mat> kernels;
        std::vector<cv::Mat> image_lap_pyr;
        std::vector<cv::Mat> foveated_pyr;
        std::vector<cv::Mat> image_sizes;
        std::vector<cv::Mat> kernel_sizes;
        
        cv::Mat image_smallest_level;
        cv::Mat down;
        cv::Mat up;           
        cv::Mat foveated_image;    
};
