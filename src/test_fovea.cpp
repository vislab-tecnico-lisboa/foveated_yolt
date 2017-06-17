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
#include <boost/algorithm/string.hpp>

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
    //const string ground_truth_labels = string(argv[14]);  // file with ground truth labels used to classification error
//    const string ground_bbox_dir = string(argv[15]);

    // Set mode
    if (strcmp(argv[6], "CPU") == 0){
        Caffe::set_mode(Caffe::CPU);
        //cout << "Using CPU\n" << endl;
    }
    else{
        Caffe::set_mode(Caffe::GPU);
        int device_id = atoi(argv[7]);
        //cout << "GPU "<< device_id << endl;
        Caffe::SetDevice(device_id);
    }


    // Load network, pre-processment, set mean and load labels
    Network Network(model_file, weight_file, mean_file, label_file);



    /**********************************************************************/
    //              LOAD LIST OF IMAGES AND BBOX OF DIRECTORY             //
    /**********************************************************************/
    std::string dir = string(argv[8]);              // directory with validation set
    std::string dir_no_blur = string(argv[15]);   // directory with images without blur


    std::vector<cv::String> files ;
    std::vector<cv::String> files_no_blur;

    files = Network.GetDir (dir, files);
    files_no_blur = Network.GetDir(dir_no_blur, files_no_blur);

    glob(dir, files);
    glob(dir_no_blur, files_no_blur);



    ofstream raw_bbox_file;

    //cv::Mat crop;
    cv::Mat foveated_image;

    raw_bbox_file.open ("raw_bbox_parse_foveal_vgg.txt",ios::app);                  // file with 5 classes + scores; 5 bounding boxes


    // FOR EACH IMAGE OF THE DATASET
    for (unsigned int input = 0;input < files_no_blur.size(); ++input){


        string file_no_blur = files_no_blur[input];


        std::vector<string> new_labels;
        std::vector<float> new_scores;

        cv::Mat img_no_blur = cv::imread(file_no_blur, 1);

        ClassData mydata(N);

        // Foveate on center of image
        resize(img_no_blur,img_no_blur, Size(size_map,size_map));
        int m = floor(4*img_no_blur.size().height);
        int n = floor(4*img_no_blur.size().width);

        img_no_blur.convertTo(img_no_blur, CV_64F);

        // Compute kernels
        std::vector<Mat> kernels = createFilterPyr(m, n, levels, sigma);

        // Construct Pyramid
        LaplacianBlending pyramid(img_no_blur,levels, kernels);


        cv::Mat center(2,1,CV_32S);
        center.at<int>(0,0) = img_no_blur.size().width;
        center.at<int>(1,0) = img_no_blur.size().height;


        // Foveate
        cv::Mat foveated_image = pyramid.foveate(center);

        foveated_image.convertTo(foveated_image,CV_8UC3);
        //imshow("fovea", foveated_image);

        //waitKey(0);


        // Predict top 5
        mydata = Network.Classify(foveated_image, N);


        std::vector<Rect> bboxes;


        raw_bbox_file << std::fixed << std::setprecision(4) << sigma << ";" << thresh << ";" ;


        // For each predicted class label:
        for (int i = 0; i < N; ++i) {

            /*******************************************/
            //  Weakly Supervised Object Localization  //
            // Saliency Map + Segmentation Mask + BBox //
            /*******************************************/
                         // Read image

            Rect Min_Rect = Network.CalcBBox(N, i, mydata, thresh);

            bboxes.push_back(Min_Rect); // save all bounding boxes

            if (i==N-1)
                raw_bbox_file << mydata.label[i] << ";" << mydata.score[i] << ";" << Min_Rect.x << ";" << Min_Rect.y << ";" << Min_Rect.width << ";" << Min_Rect.height;
            else
                raw_bbox_file << mydata.label[i] << ";" << mydata.score[i] << ";" << Min_Rect.x << ";" << Min_Rect.y << ";" << Min_Rect.width << ";" << Min_Rect.height << ";";



            }

        raw_bbox_file << endl;
    }


}


