#include <caffe/caffe.hpp>
#include <caffe/util/io.hpp>
#include <caffe/blob.hpp>
#include <caffe/layer.hpp>

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
#include "laplacian_foveation.hpp"

using namespace caffe;
using namespace std;
using std::string;


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
    static int N = atoi(argv[9]);          // define number of top predicted labels
    static float thresh = atof(argv[10]);  // segmentation threshold for mask
    static int size_map = atoi(argv[11]);  // Size of the network input images (227,227)
    static int levels = atoi(argv[12]);    // Number of kernel levels
    int sigma = atoi(argv[13]);            // Size of the fovea


    // Set mode
    if (strcmp(argv[6], "CPU") == 0){
        Caffe::set_mode(Caffe::CPU);
        //cout << "Using CPU\n" << endl;
    }
    else{
        Caffe::set_mode(Caffe::GPU);
        int device_id = atoi(argv[7]);
        Caffe::SetDevice(device_id);
    }

    // Load network, pre-processment, set mean and load labels
    Network Network(model_file, weight_file, mean_file, label_file);

    string file = string(argv[8]) + "ILSVRC2012_val_00000003.JPEG";            // load image
    //string file = string(argv[8]) + "resize_000010.jpg";
    cout << "\n------- Prediction for " << file << " ------\n" << endl;


    cv::Mat img = cv::imread(file, 1);		 // Read image

    ClassData mydata(N);

    // Predict top 5
    mydata = Network.Classify(img,N);


    /*******************************************/
    //  Weakly Supervised Object Localisation  //
    // Saliency Map + Segmentation Mask + BBox //
    /*******************************************/

    std::vector<Rect> bboxes = Network.CalcBBox(N,img, mydata, thresh);

    string input = "";
    cout << "Do you want to visualize the bounding boxes? [y/n] \n>";
    getline(cin, input);

    if (input == "y"){
        Mat copy_img;
        img.copyTo(copy_img);

        // Visualize bounding boxes on input image
        Network.VisualizeBBox(bboxes, N, copy_img, size_map);

    }


    /*******************************************************/
    //      Image Re-Classification with Attention         //
    // Foveated Image + Forward + Predict new class labels //
    /*******************************************************/


    // Foveate images
    resize(img,img, Size(size_map,size_map));
    int m = floor(4*img.size().height);
    int n = floor(4*img.size().width);

    img.convertTo(img, CV_64F);

    // Compute kernels
    std::vector<Mat> kernels = createFilterPyr(m, n, levels, sigma);

    // Construct Pyramid
    LaplacianBlending pyramid(img,levels, kernels);


    // Find Bounding Box Centroid
    for (int k=0; k<N; ++k){


        cv::Mat center(2,1,CV_32S);
        center.at<int>(0,0) = bboxes[k].y + bboxes[k].height/2;
        center.at<int>(1,0) = bboxes[k].x + bboxes[k].width/2;

        cout<<"Rectangle " <<k<< " Centroid position is at: " << center.at<int>(1,0) << " " << center.at<int>(0,0) << endl;


        // Foveate
        cv::Mat foveated_image = pyramid.foveate(center);

//        foveated_image.convertTo(foveated_image,CV_8UC3);
//        cv::resize(foveated_image,foveated_image,Size(size_map,size_map));
//        imshow("Foveada", foveated_image);
//        waitKey(0);


    }




    // Forward

    // Predict top 5 of each predicted class
    //mydata = Network.Classify(img);

    //cout << "Look Twice: \n" << mydata << endl;*/



/*********************************************************/
//               Rank Top 5 final solution               //
//      From 25 predicted labels, find highest 5         //
/*********************************************************/
// We have 25 predicted labels





}
