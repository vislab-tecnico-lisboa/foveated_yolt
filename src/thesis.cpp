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

    std::vector<cv::String> files ;

    files = Network.GetDir (dir, files);

    glob(dir, files);

//    vector<string> bbox_files ;

//    bbox_files = Network.GetDir (ground_bbox_dir, bbox_files);
//    glob(ground_bbox_dir, bbox_files);


    // File with ground truth labels
//    ifstream ground_truth_file;
//    ground_truth_file.open(ground_truth_labels.c_str());
//    string ground_class;

//    int counter_top1_yolo = files.size();
//    int counter_top5_yolo = files.size();
//    int counter_top1_yolt = files.size();
//    int counter_top5_yolt = files.size();



    ofstream raw_bbox_file;
    ofstream feedback_detection;

    raw_bbox_file.open ("raw_bbox_parse.txt",ios::app);                  // file with 5 classes + scores; 5 bounding boxes
    feedback_detection.open ("feedback_detection_parse.txt", ios::app);  // file with 25 predicted classes for each image

    raw_bbox_file <<  std::fixed << std::setprecision(4) << "sigma;" << "thres;" << "class1;" << "score1;"<< "x1;" << "y1;" << "w1;" << "h1;" << "class2;" << "score2;"<< "x2;" << "y2;" << "w2;" << "h2;";
    raw_bbox_file << "class3;" << "score3;"<< "x3;" << "y3;" << "w3;" << "h3;" << "class4;" << "score4;" << "x4;" << "y4;" << "w4;" << "h4;" << "class5;" << "score5;" << "x5;" << "y5;" << "w5;" << "h5" << "\n";

    feedback_detection << std::fixed << std::setprecision(4) << "sigma;" << "thres;" << "class1;" << "score1;" << "class2;" << "score2;" << "class3;" << "score3;" << "class4;" << "score4;" << "class5;" << "score5;" ;
    feedback_detection << "class6;" << "score6;" << "class7;" << "score7;" << "class8;" << "score8;" << "class9;" << "score9;" << "class10;" << "score10;";
    feedback_detection << "class11;" << "score11;"  << "class12;" << "score12;"  << "class13;" << "score13;"  << "class14;" << "score14;"  << "class15;" << "score15;";
    feedback_detection << "class16;" << "score16;"  << "class17;" << "score17;"  << "class18;" << "score18;"  << "class19;" << "score19;"  << "class20;" << "score20;";
    feedback_detection << "class21;" << "score21;"  << "class22;" << "score22;"  << "class23;" << "score23;"  << "class24;" << "score24;"  << "class25;" << "score25" << "\n";


    std::vector<string> new_labels;
    std::vector<float> new_scores;

    // FOR EACH IMAGE OF THE DATASET
    for (unsigned int input = 0;input < files.size(); ++input){

        string file = files[input];

//        xml_document doc;
//        string bbox_file = bbox_files[input];

//        doc.load_file("/home/filipa/PycharmProjects/C++/Foveated-YOLT/bbox/ILSVRC2012_val_00000001.xml");


//        raw_bbox_file.open ("raw_bbox_parse.txt",ios::app);                  // file with 5 classes + scores; 5 bounding boxes
//        feedback_detection.open ("feedback_detection_parse.txt", ios::app);  // file with 25 predicted classes for each image

        cv::Mat img = cv::imread(file, 1);		 // Read image
        //cv::Mat copy_img;
        //img.copyTo(copy_img);
        ClassData mydata(N);

        // Predict top 5
        mydata = Network.Classify(img, N);

        // Check if predicted labels = ground truth labels - YOLO
//        getline(ground_truth_file,ground_class);

//        if (strstr(mydata.label[0].c_str(), ground_class.c_str())){
//            counter_top1_yolo-=1; // acertou a primeira na class
//        }
//        for (int k=0; k<N; ++k){
//            if (strstr(mydata.label[k].c_str(), ground_class.c_str()))
//                counter_top5_yolo-=1; // class verdadeira esta no top
//        }
        std::vector<Rect> bboxes;
//        std::vector<string> new_labels;
//        std::vector<float> new_scores;

        raw_bbox_file << sigma << ";" << thresh << ";" ;

        feedback_detection << sigma << ";" << thresh << ";" ;

        // For each predicted class label:
        for (int i = 0; i < N; ++i) {

            /*******************************************/
            //  Weakly Supervised Object Localization  //
            // Saliency Map + Segmentation Mask + BBox //
            /*******************************************/

            Rect Min_Rect = Network.CalcBBox(N, i,img, mydata, thresh);

            bboxes.push_back(Min_Rect); // save all bounding boxes

            if (i==N-1)
                raw_bbox_file << mydata.label[i] << ";" << mydata.score[i] << ";" << Min_Rect.x << ";" << Min_Rect.y << ";" << Min_Rect.width << ";" << Min_Rect.height;
            else
                raw_bbox_file << mydata.label[i] << ";" << mydata.score[i] << ";" << Min_Rect.x << ";" << Min_Rect.y << ";" << Min_Rect.width << ";" << Min_Rect.height << ";";


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


            cv::Mat center(2,1,CV_32S);
            center.at<int>(0,0) = Min_Rect.y + Min_Rect.height/2;
            center.at<int>(1,0) = Min_Rect.x + Min_Rect.width/2;


            // Foveate
            cv::Mat foveated_image = pyramid.foveate(center);

            foveated_image.convertTo(foveated_image,CV_8UC3);
            cv::resize(foveated_image,foveated_image,Size(size_map,size_map));
//            imshow("Foveada", foveated_image);
//            waitKey(0);

            // Forward

            // Predict New top 5 of each predicted class
            ClassData feedback_data = Network.Classify(foveated_image, N);


            // For each bounding box
            for(int m=0; m<N; ++m){

                new_labels.push_back(feedback_data.label[m]);
                new_scores.push_back(feedback_data.score[m]);

                //feedback_detection <<  feedback_data.label[m] << " " << feedback_data.score[m] << " ";
            }
        }

        raw_bbox_file << endl;

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
//    output_file << sigma << " " << thresh << " " << double(counter_top1_yolo)/double(files.size());
//    output_file << " " << double(counter_top5_yolo)/double(files.size());
//    output_file <<" " << double(counter_top1_yolt)/double(files.size()) << " " << double(counter_top5_yolt)/double(files.size()) << endl;
//    output_file.close();

//    raw_bbox_file.open("raw_bbox.txt",ios::app);
//    raw_bbox_file << "\n" ;
//    raw_bbox_file.close();

}


