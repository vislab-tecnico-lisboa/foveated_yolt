#include <caffe/caffe.hpp>
#include <caffe/util/io.hpp>
#include <caffe/blob.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <vector>
#include <memory>
#include <math.h>
#include <limits>





#include "network_classes.hpp"

//#include <boost/shared_ptr.hpp>
//#include <stdio>

using namespace caffe;
using namespace std;
using std::string;
//using cv::Mat;
//using namespace boost::numpy;

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;

/*****************************************/
//		MAIN
/*****************************************/

int main(int argc, char** argv){

    // Init
    ::google::InitGoogleLogging(argv[0]);

    const string absolute_path_folder = string(argv[1]);
    const string model_file = absolute_path_folder + string(argv[2]);
    const string weight_file = absolute_path_folder + string(argv[3]);
    const string mean_file = absolute_path_folder + string(argv[4]);
    const string label_file = absolute_path_folder + string(argv[5]);


    // Set mode
    if (strcmp(argv[6], "CPU") == 0){
        Caffe::set_mode(Caffe::CPU);
        //cout << "Using CPU\n" << endl;
    }
    else{
        Caffe::set_mode(Caffe::GPU);
        int device_id = atoi(argv[7]);
        Caffe::SetDevice(device_id);
        //cout << "Using GPU, device_id\n" << device_id << "\n" << endl;
    }

    // Load network, pre-processment, set mean and load labels
    Network Network(model_file, weight_file, mean_file, label_file);

    string file = string(argv[8]) + "ILSVRC2012_val_00000001.JPEG";            // load image
    //string file = string(argv[8]) + "resize_000010.jpg";
    cout << "\n------- Prediction for " << file << " ------\n" << endl;

    cv::Mat img = cv::imread(file, -1);		 // Read image
    static int N = 5;
    ClassData mydata(N);

    // Predict top 5
    mydata = Network.Classify(img);


    /***************************************/
    // Weakly Supervised Object Localisation
    /***************************************/

    Network.BackwardPass(N,img, mydata);

}
