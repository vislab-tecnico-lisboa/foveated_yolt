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

const int CARTESIAN=1;
const int FOVEATION=2;
const int HYBRID=3;

using namespace caffe;
using namespace std;

using std::string;


/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;


cv::Mat foveate(const cv::Mat & img, const int & size_map, const int & levels, const int & sigma, const cv::Mat & fixation_point)
{
    // Foveate images
    int m = floor(4*img.size().height);
    int n = floor(4*img.size().width);

    img.convertTo(img, CV_64F);

    // Compute kernels
    std::vector<Mat> kernels = createFilterPyr(m, n, levels, sigma);

    // Construct Pyramid
    LaplacianBlending pyramid(img,levels, kernels);

    // Foveate
    cv::Mat foveated_image = pyramid.foveate(fixation_point);

    foveated_image.convertTo(foveated_image,CV_8UC3);
    cv::resize(foveated_image,foveated_image,Size(size_map,size_map));

    return foveated_image;
}

/*****************************************/
//		MAIN
/*****************************************/

int main(int argc, char** argv) {

    // Init
    ::google::InitGoogleLogging(argv[0]);

    /* network params */
    const string absolute_path_folder = string(argv[1]);
    const string model_file = absolute_path_folder + string(argv[2]);
    const string weight_file = absolute_path_folder + string(argv[3]);
    const string mean_file = absolute_path_folder + string(argv[4]);
    const string label_file = absolute_path_folder + string(argv[5]);

    /* method specific params */
    static int N = atoi(argv[9]);          // define number of top predicted labels
    static float thresh = atof(argv[10]);  // segmentation threshold for mask
    static int size_map = atoi(argv[11]);  // Size of the network input images (227,227)
    static int levels = atoi(argv[12]);    // Number of kernel levels
    int sigma = atoi(argv[13]);            // Size of the fovea
    static int mode = atoi(argv[14]);    // Number of kernel levels

    // Set mode
    if (strcmp(argv[6], "CPU") == 0){
        Caffe::set_mode(Caffe::CPU);
    }
    else{
        Caffe::set_mode(Caffe::GPU);
        int device_id = atoi(argv[7]);
        Caffe::SetDevice(device_id);
    }


    // Load network, pre-processment, set mean and load labels
    Network Network(model_file, weight_file, mean_file, label_file);

    /**********************************************************************/
    //              LOAD LIST OF IMAGES AND BBOX OF DIRECTORY             //
    /**********************************************************************/

    std::string dir = string(argv[8]);              // directory with validation set

    std::vector<cv::String> image_image_files ;

    image_image_files = Network.GetDir (dir, image_image_files);

    glob(dir, image_image_files);

    ofstream feedforward_detection;
    ofstream feedback_detection;

    // store results
    feedforward_detection.open ("feedforward_detection_parse.txt",ios::app);          // file with 5 classes + scores + 5 bounding boxes
    feedback_detection.open ("feedback_detection_parse.txt", ios::app);  // file with 25 predicted classes for each image

    // FOR EACH IMAGE OF THE DATASET (TODO: OPTIMIZATION -> PROCESS BATCH OF IMAGES INSTEAD OF SINGLE IMAGES)
    for (unsigned int input = 0;input < image_image_files.size(); ++input){

        string file = image_image_files[input];
        std::vector<string> new_labels;
        std::vector<float> new_scores;
        std::vector<Rect> bboxes;

        cv::Mat img = cv::imread(file, 1);		 // Read image
        resize(img,img, Size(size_map,size_map)); // Resize to network size
        cv::Mat img_orig=img.clone();

        ClassData mydata(N);


        if(mode==FOVEATION)
        {
            cv::Mat fixation_point(2,1,CV_32S);
            fixation_point.at<int>(0,0) = img.size().width*0.5;
            fixation_point.at<int>(1,0) = img.size().height*0.5;
            img=foveate(img,size_map,levels,sigma,fixation_point);
        }

        // FEEDFORWAD - PREDICT CLASSES (TOP N)
        mydata = Network.Classify(img, N);



        // store results
        feedforward_detection << std::fixed << std::setprecision(4) << sigma << ";" << thresh << ";";
        feedback_detection <<  std::fixed << std::setprecision(4) << sigma << ";" << thresh << ";";

        // For each predicted class label:
        for (int i = 0; i < N; ++i) {

            /*******************************************/
            //  Weakly Supervised Object Localization  //
            // Saliency Map + Segmentation Mask + BBox //
            /*******************************************/
            Rect Min_Rect = Network.CalcBBox(N, i,img, mydata, thresh);
            bboxes.push_back(Min_Rect); // save all bounding boxes

            // store results
            if (i==N-1)
                feedforward_detection << mydata.label[i] << ";" << mydata.score[i] << ";" << Min_Rect.x << ";" << Min_Rect.y << ";" << Min_Rect.width << ";" << Min_Rect.height;
            else
                feedforward_detection << mydata.label[i] << ";" << mydata.score[i] << ";" << Min_Rect.x << ";" << Min_Rect.y << ";" << Min_Rect.width << ";" << Min_Rect.height << ";";


            /*******************************************************/
            //      Image Re-Classification with Attention         //
            // Foveated Image + Forward + Predict new class labels //
            /*******************************************************/

            if(mode==FOVEATION||mode==HYBRID)
            {
                cv::Mat fixation_point(2,1,CV_32S);
                fixation_point.at<int>(0,0) = Min_Rect.y + Min_Rect.height/2;
                fixation_point.at<int>(1,0) = Min_Rect.x + Min_Rect.width/2;
                img=foveate(img_orig,size_map,levels,sigma,fixation_point);
            }
            else
            {
                /*****************************************/
                            // CROP IMAGE:

                cv::Mat crop = img_orig(bboxes[i]);  // crop image by bbox


                if (crop.size().width != 0 && crop.size().height != 0 )
                    cv::resize(crop,crop,Size(size_map,size_map));
                else
                    img_orig.copyTo(crop);
            }
            // Forward

            // Predict New top 5 of each predicted class
            ClassData feedback_data = Network.Classify(img, N);


            // For each bounding box
            for(int m=0; m<N; ++m){
                new_labels.push_back(feedback_data.label[m]);
                new_scores.push_back(feedback_data.score[m]);

                //feedback_detection <<  feedback_data.label[m] << " " << feedback_data.score[m] << " ";
            }
        }

        feedforward_detection << endl;

        for (int aux=0; aux<N*N; ++aux){
            if (aux==N*N-1)
                feedback_detection <<  new_labels[aux] << ";" << new_scores[aux];
            else
                feedback_detection <<  new_labels[aux] << ";" << new_scores[aux] << ";";
        }

     }
     feedback_detection << endl;


        /*********************************************************/
        //               Rank Top 5 final solution               //
        //      From 25 predicted labels, find highest 5         //
        /*********************************************************/

        //for (int k=0; k<new_labels.size(); ++k)
        //    cout << "New Prediction: " << new_labels[k] << "\t\t" << new_scores[k] << endl;

//        std::vector<string> top_final_labels;
//        std::vector<float> top_final_scores;


//        std::vector<int> topN = Argmax(new_scores, N);

//        for (int top = 0; top <N; ++top){
//            int idx = topN[top];

//            top_final_labels.push_back(new_labels[idx]);
//            top_final_scores.push_back(new_scores[idx]);
//        }

//        cout << "Final Solution:\n " << endl;
//        for (int top = 0; top <N; ++top)
//            cout << "Score: " << top_final_scores[top]  << "\t Label: " << top_final_labels[top] << endl;


        // Check if predicted labels = ground truth labels - YOLT
//        if (strstr(top_final_labels[0].c_str(), ground_class.c_str())){
//            counter_top1_yolt-=1; // acertou a primeira na class
//        }
//        for (int k=0; k<N; ++k){
//            if (strstr(top_final_labels[k].c_str(), ground_class.c_str()))
//                counter_top5_yolt-=1; // class verdadeira esta no top
//                break;
//        }


//        /***************************************************************/
//        //                  LOCALIZATION ERROR                         //
//        /***************************************************************/

//        xpath_node_set get_width = doc.select_nodes("/annotation/size/width");
//        xpath_node_set get_height = doc.select_nodes("/annotation/size/height");

//        for (xpath_node_set::const_iterator it = get_width.begin(); it!= get_width.end(); ++it){
//            xpath_node node_width = *it;
//            cout << "name " << node_width.node().name() << endl;
//            cout << "Width " << node_width.node().child_value() << endl;
//            string img_width = node_width.node().child_value();
//        }
//        for (xpath_node_set::const_iterator it = get_height.begin(); it!= get_height.end(); ++it){
//            xpath_node node_height = *it;
//            cout << "name " << node_height.node().name() << endl;
//            cout << "Height " << node_height.node().child_value() << endl;
//            string img_height = node_height.node().child_value();
//        }


 //   }

    //ground_truth_file.close();




    /*********************************************************/
    //           WRITE OUTPUT FILE WITH THE RESULTS          //
    /*********************************************************/

//    // Write data to output file
//    ofstream output_file;
//    output_file.open ("final_results.txt",ios::app);  //appending the content to the current content of the file.
//    output_file << std::fixed << std::setprecision(4) ;
//    output_file << sigma << " " << thresh << " " << double(counter_top1_yolo)/double(image_files.size());
//    output_file << " " << double(counter_top5_yolo)/double(image_files.size());
//    output_file <<" " << double(counter_top1_yolt)/double(image_files.size()) << " " << double(counter_top5_yolt)/double(image_files.size()) << endl;
//    output_file.close();

//    feedforward_detection.open("raw_bbox.txt",ios::app);
//    feedforward_detection << "\n" ;
//    feedforward_detection.close();

}


