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
    std::string dir = string(argv[8]);              // directory with validation set
    std::string dir_no_blur = string(argv[15]);   // directory with images without blur


    std::vector<cv::String> files ;
    std::vector<cv::String> files_no_blur;

    files = Network.GetDir (dir, files);
    files_no_blur = Network.GetDir(dir_no_blur, files_no_blur);

    glob(dir, files);
    glob(dir_no_blur, files_no_blur);



    ofstream raw_bbox_file;
    ofstream feedback_detection;
    //cv::Mat crop;
    cv::Mat foveated_image;


    // FOR EACH IMAGE OF THE DATASET
    for (unsigned int input = 0;input < files.size(); ++input){

        string file = files[input];
        string file_no_blur = files_no_blur[input];

         cout << file << endl;
          cout << file_no_blur << endl;

        std::vector<string> new_labels;
        std::vector<float> new_scores;

        cv::Mat img = cv::imread(file, 1);		 // Read image


        ClassData mydata(N);

        // Predict top 5
        mydata = Network.Classify(img, N);

        std::vector<Rect> bboxes;


        // For each predicted class label:
        for (int i = 0; i < N; ++i) {

            /*******************************************/
            //  Weakly Supervised Object Localization  //
            // Saliency Map + Segmentation Mask + BBox //
            /*******************************************/
            cv::Mat img_no_blur = cv::imread(file_no_blur, 1);		 // Read image
            cv::Mat aux;
            img_no_blur.copyTo(aux);

            Rect Min_Rect = Network.CalcBBox(N, i, mydata, thresh);

            bboxes.push_back(Min_Rect); // save all bounding boxes

            resize(aux,aux, Size(size_map,size_map));
            rectangle(aux, bboxes[i], Scalar(0, 0, 255), 2, 8, 0 );
            imwrite("local_puma_0.png",aux);
            imshow("local", aux);
            waitKey(0);


           // cout << Min_Rect.x << " " << Min_Rect.y << " " << Min_Rect.width << " " << Min_Rect.height << endl;

            /*******************************************************/
            //      Image Re-Classification with Attention         //
            // Foveated Image + Forward + Predict new class labels //
            /*******************************************************/

              /*****************************************/
            // FOVEATE IMAGES:
            resize(img_no_blur,img_no_blur, Size(size_map,size_map));
            int m = floor(4*img.size().height);
            int n = floor(4*img.size().width);

            img_no_blur.convertTo(img_no_blur, CV_64F);

            // Compute kernels
            std::vector<Mat> kernels = createFilterPyr(m, n, levels, sigma);

            // Construct Pyramid
            LaplacianBlending pyramid(img_no_blur,levels, kernels);


            cv::Mat center(2,1,CV_32S);
            center.at<int>(0,0) = Min_Rect.y + Min_Rect.height/2;
            center.at<int>(1,0) = Min_Rect.x + Min_Rect.width/2;


            // Foveate
            cv::Mat foveated_image = pyramid.foveate(center);

           // foveated_image.convertTo(foveated_image,CV_8UC3);
            cv::resize(foveated_image,foveated_image,Size(size_map,size_map));
//            circle(foveated_image, Point(center.at<int>(0,0), center.at<int>(1,0)), 40,Scalar(0, 0, 255), 2, 8, 0 );
//            imwrite("foveated_puma_1.png",foveated_image);
             foveated_image.convertTo(foveated_image,CV_8UC3);

//            imshow("Foveada", foveated_image);
//            waitKey(0);




            // Forward
            // Predict New top 5 of each predicted class
            ClassData feedback_data = Network.Classify(foveated_image, N);  // foveated_image  or crop

            // For each bounding box
            for(int m=0; m<N; ++m){

                new_labels.push_back(feedback_data.label[m]);
                new_scores.push_back(feedback_data.score[m]);

                //cout <<  feedback_data.label[m] << " " << feedback_data.score[m] << endl;
            }
        }



        ClassData top_feedback_data(N);
        std::vector<int> topN = Argmax(new_scores, N);

        for (int i = 0; i < N; ++i) {
            int idx = topN[i];

            top_feedback_data.index[i] = idx;
            top_feedback_data.label[i] = new_labels[idx];
            top_feedback_data.score[i] = new_scores[idx];
            //cout << "top label " << top_feedback_data.label[i] <<  " " <<  top_feedback_data.score[i] << endl;

        }




        // For each top final predicted label
        for (int j = 0; j < N; ++j) {

            Rect Min_Feedback_Rect = Network.CalcBBox(N, j, top_feedback_data, thresh);

            rectangle(img, Min_Feedback_Rect, Scalar(0, 0, 255), 2, 8, 0 );
//            img.convertTo(img,CV_8UC3);
//            imshow("bbox after crop", img);
//            waitKey(0);


        }
}


}


