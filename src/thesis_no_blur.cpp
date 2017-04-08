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
    std::string dir = string(argv[8]);              // directory with images with blur

    std::string dir_no_blur = string(argv[15]);   // directory with images without blur

    std::vector<cv::String> files ;
    std::vector<cv::String> files_no_blur;

    files = Network.GetDir (dir, files);
    files_no_blur = Network.GetDir(dir_no_blur, files_no_blur);

    glob(dir, files);
    glob(dir_no_blur, files_no_blur);



    ofstream raw_bbox_file;
    ofstream feedback_detection;


    raw_bbox_file.open ("raw_bbox_parse_high_blur_caffenet_100.txt",ios::app);                  // file with 5 classes + scores; 5 bounding boxes
    feedback_detection.open ("feedback_detection_parse_high_blur_caffenet_100.txt", ios::app);  // file with 25 predicted classes for each image

    // FOR EACH IMAGE OF THE DATASET
    for (unsigned int input = 0;input < files.size(); ++input){

        string file = files[input];
        string file_no_blur = files_no_blur[input];


        std::vector<string> new_labels;
        std::vector<float> new_scores;

        cv::Mat img = cv::imread(file, 1);		 // Read image

        ClassData mydata(N);

        // Predict top 5
        mydata = Network.Classify(img, N);

        raw_bbox_file << std::fixed << std::setprecision(4) << sigma << ";" << thresh << ";" ;

        feedback_detection <<  std::fixed << std::setprecision(4) << sigma << ";" << thresh << ";" ;

       
        // For each predicted class label:
        for (int i = 0; i < N; ++i) {

            /*******************************************/
            //  Weakly Supervised Object Localization  //
            // Saliency Map + Segmentation Mask + BBox //
            /*******************************************/

            cv::Mat img_no_blur = cv::imread(file_no_blur, 1);		 // Read image

            if (i==N-1)
                raw_bbox_file << mydata.label[i] << ";" << mydata.score[i] ;
            else
                raw_bbox_file << mydata.label[i] << ";" << mydata.score[i] << ";" ;

            // Forward
            // Predict New top 5 of each predicted class
            ClassData feedback_data = Network.Classify(img_no_blur, N);  // foveated_image  or crop

            // For each bounding box
            for(int m=0; m<N; ++m){

                new_labels.push_back(feedback_data.label[m]);
                new_scores.push_back(feedback_data.score[m]);

            }
        }

        raw_bbox_file << endl;

        for (int aux=0; aux<N*N; ++aux){
            if (aux==N*N-1)
                feedback_detection <<  new_labels[aux] << ";" << new_scores[aux] << endl;
            else
            	feedback_detection <<  new_labels[aux] << ";" << new_scores[aux] << ";";
        }

       
    }
   // feedback_detection << endl;

}



