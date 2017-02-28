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
    const string ground_truth_labels = string(argv[14]);  // file with ground truth labels used to classification error


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


    /**********************************************************************/
    //                  LOAD LIST OF IMAGES OF DIRECTORY                  //
    /**********************************************************************/
    string dir = string(argv[8]);              // directory with validation set
    cout << "dir " << dir << endl;
    vector<string> files ;

    files = Network.GetDir (dir, files);
    glob(dir, files);

//    cout << "file " << ground_truth_labels << endl;

    // File with ground truth labels
    ifstream ground_truth_file;
    ground_truth_file.open("/home/filipa/PycharmProjects/C++/Foveated-YOLT/files/ground_truth_labels_ilsvrc12.txt");
    string ground_class;

    int counter_top1_yolo = files.size();
    int counter_top5_yolo = files.size();
    int counter_top1_yolt = files.size();
    int counter_top5_yolt = files.size();


    // FOR EACH IMAGE OF THE DATASET
    for (unsigned int input = 0;input < files.size(); ++input){

        string file = files[input];

        cv::Mat img = cv::imread(file, 1);		 // Read image

        ClassData mydata(N);


        // Predict top 5
        mydata = Network.Classify(img, N);


        getline(ground_truth_file,ground_class);
//        cout << "ground label " << ground_class << endl;

        if (strstr(mydata.label[0].c_str(), ground_class.c_str())){
            counter_top1_yolo-=1; // acertou a primeira na class
        }
        for (int k=0; k<N; ++k){
            if (strstr(mydata.label[k].c_str(), ground_class.c_str()))
                counter_top5_yolo-=1; // class verdadeira esta no top
        }


//        cout << "top1 " << counter_top1_yolo << " top5 " << counter_top5_yolo << endl;

        std::vector<Rect> bboxes;
        std::vector<string> new_labels;
        std::vector<float> new_scores;

        // For each predicted class label:
        for (int i = 0; i < N; ++i) {

            /*******************************************/
            //  Weakly Supervised Object Localization  //
            // Saliency Map + Segmentation Mask + BBox //
            /*******************************************/

            Rect Min_Rect = Network.CalcBBox(N, i,img, mydata, thresh);

            bboxes.push_back(Min_Rect); // save all bounding boxes

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
            //for (int k=0; k<N; ++k){


            cv::Mat center(2,1,CV_32S);
            center.at<int>(0,0) = Min_Rect.y + Min_Rect.height/2;
            center.at<int>(1,0) = Min_Rect.x + Min_Rect.width/2;

            //cout<<"Rectangle " <<k<< " Centroid position is at: " << center.at<int>(1,0) << " " << center.at<int>(0,0) << endl;


            // Foveate
            cv::Mat foveated_image = pyramid.foveate(center);

//            foveated_image.convertTo(foveated_image,CV_8UC3);
//            cv::resize(foveated_image,foveated_image,Size(size_map,size_map));
//            imshow("Foveada", foveated_image);
//            waitKey(0);

            // Forward

            // Predict New top 5 of each predicted class
            mydata = Network.Classify(foveated_image, N);



            for(int m=0; m<N; ++m){

                new_labels.push_back(mydata.label[m]);
                new_scores.push_back(mydata.score[m]);
            }


        }

        /*********************************************************/
        //               Rank Top 5 final solution               //
        //      From 25 predicted labels, find highest 5         //
        /*********************************************************/

        //for (int k=0; k<new_labels.size(); ++k)
        //    cout << "New Prediction: " << new_labels[k] << "\t\t" << new_scores[k] << endl;

        std::vector<string> top_final_labels;
        std::vector<float> top_final_scores;


        std::vector<int> topN = Argmax(new_scores, N);

        for (int top = 0; top <N; ++top){
            int idx = topN[top];

            top_final_labels.push_back(new_labels[idx]);
            top_final_scores.push_back(new_scores[idx]);
        }

//        cout << "Final Solution:\n " << endl;
//        for (int top = 0; top <N; ++top)
//            cout << "Score: " << top_final_scores[top]  << "\t Label: " << top_final_labels[top] << endl;



        if (strstr(top_final_labels[0].c_str(), ground_class.c_str())){
            counter_top1_yolt-=1; // acertou a primeira na class
        }
        for (int k=0; k<N; ++k){
            if (strstr(top_final_labels[k].c_str(), ground_class.c_str()))
                counter_top5_yolt-=1; // class verdadeira esta no top
                break;
        }


//        cout << "top1 " << counter_top1_yolt << " top5 " << counter_top5_yolt << endl;


    /*********************************************************/
    //             COMPUTE CLASSIFICATION ERROR              //
    /*********************************************************/




    }


    ground_truth_file.close();


    // Write data to output file
    ofstream output_file;
    output_file.open ("final_results.txt",ios::app);  //appending the content to the current content of the file.
    output_file << std::fixed << std::setprecision(4) ;
    output_file << "Sigma: " << sigma << " Thres: " << thresh << " Top1: " << double(counter_top1_yolo)/double(files.size());
    output_file << " Top5 " << double(counter_top5_yolo)/double(files.size());
    output_file <<" Top1: " << double(counter_top1_yolt)/double(files.size()) << " Top5 " << double(counter_top5_yolt)/double(files.size()) << endl;
    output_file.close();

}























//    string input = "";
//    cout << "Do you want to visualize the bounding boxes? [y/n] \n>";
//    getline(cin, input);

//    if (input == "y"){
//        Mat copy_img;
//        //img.copyTo(copy_img);

//        // Visualize bounding boxes on input image
//        Network.VisualizeBBox(bboxes, N, img, size_map);

//    }
