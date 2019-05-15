﻿#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <dirent.h>

#define CV_64FC3 CV_MAKETYPE(CV_64F,3)

using namespace caffe;
using namespace std;
using namespace cv;


class ClassData {

    public:
        ClassData(int N_);
        ClassData(std::vector<string> label_, std::vector<float> score_, std::vector<int> index_);
        ~ClassData();

        friend ostream& operator<<(ostream& output, const ClassData& D);  
        int N;
        std::vector<string> label;
        std::vector<float> score;
        std::vector<int> index;


};


// Callback for function std::partial_sort used in ArgMax
bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs);
std::vector<int> ArgMax(const std::vector<float>& v, int n);


class Network {

    public:
        Network(const string& model_file, const string& weight_file,
                const string& mean_file,  const string& label_file); //construtor

        // Return Top 5 prediction of image in mydata
        ClassData Classify(const cv::Mat& img, int N);
        Rect CalcBBox(int class_index, ClassData mydata, float thresh,cv::Mat & saliency_map); // NEW
        void VisualizeBBox(std::vector<Rect> bboxes, int N, cv::Mat &img, int size_map, int ct, const std::string & name);
        void VisualizeFoveation(const cv::Mat & fix_pt, const cv::Mat & img, const int & sigma, const int & k, const std::string & name);
        void VisualizeSaliencyMap(const cv::Mat & saliency_map, int k, const std::string & name);
        std::vector<String> GetDir(string dir, vector<String> &files);

        float* LimitValues(float* bottom_data); // NEW
        //float find_max(Mat gradient_values);
        cv::Mat CalcRGBmax(cv::Mat i_RGB);

    private:
        void SetMean(const string& mean_file);
        void WrapInputLayer(std::vector<cv::Mat>* input_channels);
        void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);
        std::vector<float> Predict(const cv::Mat& img);

        int num_channels;
        boost::shared_ptr<caffe::Net<float> > net;

        cv::Mat mean_;
        std::vector<string> labels;
        cv::Size input_geometry;        // size of network - width and height
};

