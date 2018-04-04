#include <opencv2/opencv.hpp>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <vector>
#include <memory>
#include <math.h>
#include <limits>
#include <time.h>

#include "network_classes.hpp"
#include "laplacian_foveation.hpp"

using namespace caffe;
using namespace std;
using std::string;


cv::Mat foveate(const cv::Mat &img, const int &size_map, 
				const int &levels, const int &sigma, 
				const cv::Mat &fixation_point);

std::vector<cv::Mat> FixationPoints (int img_size, int n_points, int random);

void VisualizeFixationPoints(int img_size, std::vector<cv::Mat> fixation_points);


////////////////////
//      MAIN      //
////////////////////


int main(int argc, char** argv){

    // Init
    ::google::InitGoogleLogging(argv[0]);

    /* network params */
    const string absolute_path_folder = string(argv[1]);
    const string model_file 		  = absolute_path_folder + string(argv[2]);
    const string weight_file          = absolute_path_folder + string(argv[3]);
    const string mean_file            = absolute_path_folder + string(argv[4]);
    const string label_file           = absolute_path_folder + string(argv[5]);
    const string dataset_folder       = string(argv[6]);
    static int N                      = atoi(argv[7]);			// Define number of top predicted labels
    static string threshs_            = string(argv[8]);		// Segmentation threshold for mask
    static int size_map 			  = atoi(argv[9]);          // Size of the network input images (227,227)
    static int levels 				  = atoi(argv[10]);         // Number of kernel levels
    static string sigmas_             = string(argv[11]);       // Size of the fovea
    static string results_folder      = string(argv[12]);       // Folder to store results
    static int mode                   = atoi(argv[13]);         // Mode
    static bool debug                 = atoi(argv[14]);         // Set debug = 1 to see figures
    static int total_images           = atoi(argv[15]);         // Number of images
    if (strcmp(argv[16], "CPU") == 0)                           // Set Mode
        Caffe::set_mode(Caffe::CPU);
    else {
        Caffe::set_mode(Caffe::GPU);
        int device_id = atoi(argv[17]);
        Caffe::SetDevice(device_id);
    }
    static int npoints                 = atoi(argv[18]);         // Number of fixation points

    std::cout << "Absolute Path Folder: " 	<< absolute_path_folder<< std::endl;    
    std::cout << "Model File: " 			<< model_file << std::endl;
    std::cout << "Weights File: " 			<< weight_file << std::endl;
    std::cout << "Mean File: " 				<< mean_file<< std::endl; 
    std::cout << "Labels File: " 			<< label_file<< std::endl; 
    std::cout << "Dataset Folder: " 		<< dataset_folder<< std::endl;
    std::cout << "Results Folder: "			<< results_folder << std::endl;
    std::cout << "Top Classes: " 			<< N << std::endl;
    std::cout << "Levels: " 				<< levels << std::endl;
    std::cout << "Sigmas: " 				<< sigmas_ << std::endl;
    std::cout << "Threshs: "				<< threshs_<< std::endl;
    std::cout << "Size Map: " 				<< size_map << std::endl;
    std::cout << "Mode: " 					<< mode << std::endl;
    std::cout << "Fixation Pts: "           << npoints << std::endl;
    std::cout << "Total_images: " 			<< total_images << std::endl;
    std::cout << "Debug: "                  << debug << std::endl;

    std::vector<float> threshs;
    std::stringstream ss_(threshs_);
    float iter_th; 
    while (ss_ >> iter_th) {
        threshs.push_back(iter_th);
        if (ss_.peek() == ',')
            ss_.ignore();
    }

    std::vector<int> sigmas;
    std::stringstream ss(sigmas_);
    int iter_sig;
    while (ss >> iter_sig) {
        sigmas.push_back(iter_sig);
        if (ss.peek() == ',')
            ss.ignore();
    }

    // Load network, pre-processment, set mean and load labels
    Network Network(model_file, weight_file, mean_file, label_file);

    // Load List of images
    std::vector<cv::String> image_image_files ;
    image_image_files = Network.GetDir (dataset_folder, image_image_files);

    if(total_images>image_image_files.size())
        total_images=image_image_files.size();
    glob(dataset_folder, image_image_files);

    // store results
    ofstream feedforward_detection;
    ofstream feedback_detection;

    // File with 5 classes + scores + 5 bounding boxes
    std::string feedforward_detection_str=results_folder+string("feedforward_detection_parse.txt");
    feedforward_detection.open (feedforward_detection_str.c_str(),ios::out);
    feedforward_detection<<"sigma;thres;pt_w;pt_h;class1;score1;x1;y1;w1;h1;class2;score2;x2;y2;w2;h2;class3;score3;x3;y3;w3;h3;class4;score4;x4;y4;w4;h4;class5;score5;x5;y5;w5;h5"<<std::endl;

	// File with 25 predicted classes + scores for each image
    std::string feedback_detection_str=results_folder+string("feedback_detection_parse.txt");
    feedback_detection.open (feedback_detection_str.c_str(), ios::out);  
    feedback_detection<<"sigma;thres;pt_w;pt_h;class1;score1;class2;score2;class3;score3;class4;score4;class5;score5;class6;score6;class7;score7;class8;score8;class9;score9;class10;score10;class11;score11;class12;score12;class13;score13;class14;score14;class15;score15;class16;score16;class17;score17;class18;score18;class19;score19;class20;score20;class21;score21;class22;score22;class23;score23;class24;score24;class25;score25"<<std::endl;

    // Seed for random fixation points
    srand (time(NULL));
	std::vector<cv::Mat> fixedpts = FixationPoints(size_map,npoints,1);
    //VisualizeFixationPoints(size_map,fixedpts) ;
    
    // Total number of iterations
    int total_iterations=total_images*threshs.size()*sigmas.size()*fixedpts.size();

    float thresh;
    int sigma;
    cv::Mat fixedpt;

    // For each threshold
    for (unsigned int thresh_index=0; thresh_index<threshs.size(); ++thresh_index){
      	thresh=threshs[thresh_index];
        	
      	// For each sigma
       	for (unsigned int sigma_index = 0;sigma_index < sigmas.size(); ++sigma_index){
	        sigma=sigmas[sigma_index];

	        // For each fixation point
		    for (unsigned int fixedpt_index = 0; fixedpt_index < fixedpts.size(); ++fixedpt_index){
	            fixedpt = fixedpts[fixedpt_index];    
	    		
	    		// For each image
	    		for (unsigned int input=0; input<total_images; ++input){
	        
			        // Preprocess each image
			        string file = image_image_files[input];
			        cv::Mat img = cv::imread(file, 1);        // Read image
			        resize(img,img, Size(size_map,size_map)); // Resize to network size
			        cv::Mat img_orig=img.clone();

			        // Varibles
			        //std::vector<string> new_labels;
		            //std::vector<float> new_scores;
		            std::vector<Rect> bboxes;

		         	// Atual iteration
		            int iteration=input+1+fixedpt_index*total_images+sigma_index*total_images*fixedpts.size()+thresh_index*total_images*sigmas.size()*fixedpts.size();
		            std::cout << "Th:" << thresh
		                      << "\tSig:" << sigma 
		                      << "\tPt: (" << fixedpt.at<int>(0,0)<<","<<fixedpt.at<int>(1,0)<<')' 
		                      << "\tImg:" << input+1 
		                      << " of " << total_images 
		                      << "\tIter: " << iteration
		                      << " of "  << total_iterations 
		                      << " ("<< 100.0*(iteration)/(total_iterations) << "%)"<< std::endl;

		            // Fixation Point - Center 
		            //cv::Mat fixation_point(2,1,CV_32S);
		            //fixation_point.at<int>(0,0) = img.size().width*0.5;
		            //fixation_point.at<int>(1,0) = img.size().height*0.5;

		            // 1st pass foveation
		 			img=foveate(img,size_map,levels,sigma,fixedpt);
		 			cv::Mat img_first_pass=img.clone();

		 			// Feedfoward - Prediciton of TOP N classes
		            ClassData first_pass_data = Network.Classify(img, N);

		            // Store results
                    feedforward_detection << std::fixed << std::setprecision(4) << sigma << ";" << thresh << ";" << fixedpt.at<int>(0,0) << ";" << fixedpt.at<int>(1,0) << ";";
                    feedback_detection    << std::fixed << std::setprecision(4) << sigma << ";" << thresh << ";" << fixedpt.at<int>(0,0) << ";" << fixedpt.at<int>(1,0) << ";";


		            // For each predicted class
		            for (int class_index = 0; class_index < N; ++class_index) {
		                cv::Mat img_second_pass;

		                /////////////////////////////////////////////
		                //  Weakly Supervised Object Localization  //
		                // Saliency Map + Segmentation Mask + BBox //
		                /////////////////////////////////////////////

		                Rect Min_Rect = Network.CalcBBox(class_index,img, first_pass_data, thresh);
		               
		              	// Save all bounding boxes
		                bboxes.push_back(Min_Rect); 

		                // Store results
		                if (class_index==N-1) {
		                    feedforward_detection << first_pass_data.label[class_index] << ";" << first_pass_data.score[class_index] << ";" 
		                						  << Min_Rect.x << ";" << Min_Rect.y << ";" 
		                						  << Min_Rect.width << ";" << Min_Rect.height;
                            feedforward_detection << endl;
		                } 
                        else {
		                    feedforward_detection << first_pass_data.label[class_index] << ";" << first_pass_data.score[class_index] << ";"
		                    					  << Min_Rect.x << ";" << Min_Rect.y << ";" 
		                    					  << Min_Rect.width << ";" << Min_Rect.height << ";";
		                }

		                /////////////////////////////////////////////////////////
		                //      Image Re-Classification with Attention         //
		                // Foveated Image + Forward + Predict new class labels //
		                /////////////////////////////////////////////////////////

		                cv::Mat fixation_point(2,1,CV_32S);
		                fixation_point.at<int>(0,0) = Min_Rect.y + Min_Rect.height/2;
		                fixation_point.at<int>(1,0) = Min_Rect.x + Min_Rect.width/2;
		                img_second_pass=foveate(img_orig,size_map,levels,sigma,fixation_point);

		                // Forward
		                // Predict New top 5 of each predicted class
		                ClassData feedback_data = Network.Classify(img_second_pass, N);

		                // For each bounding box
		                for(int m=0; m<N; ++m) {
		                    //new_labels.push_back(feedback_data.label[m]);
		                    //new_scores.push_back(feedback_data.score[m]);	            

		                    // Store Feedback results
		                	if ((class_index+1)*(m+1) == N*N) {
		                    	feedback_detection <<  feedback_data.label[m] << ";" << feedback_data.score[m];
                                feedback_detection << endl;
                            }
		                	else
		                    	feedback_detection <<  feedback_data.label[m] << ";" << feedback_data.score[m] << ";";    
		                }

		                if(debug) {
		                    Mat dst;
		                    cv::hconcat(img_orig,img_first_pass, dst); // horizontal
		                    cv::hconcat(dst, img_second_pass, dst); // horizontal
		                    //cv::vconcat(a, b, dst); // vertical
		                    namedWindow( "original image,    first pass,   second pass     class", WINDOW_AUTOSIZE ); // Create a window for display.
		                    imshow( "original image,    first pass,   second pass     class", dst );                  // Show our image inside it.
		                    waitKey(1);
		                }
					}
		        }
      		}
      	}
    }
	feedforward_detection.close();
    feedback_detection.close();
}



//////////////////////
// FOVEATE FUNTION  //
//////////////////////

cv::Mat foveate(const cv::Mat &img, const int &size_map, 
				const int &levels, const int &sigma, 
				const cv::Mat &fixation_point) {

    cv::Mat image;
    img.convertTo(image, CV_64F);
    
    // Construct Pyramid
    LaplacianBlending pyramid(image,levels, sigma);

    // Foveate
    cv::Mat foveated_image = pyramid.Foveate(fixation_point);

    foveated_image.convertTo(foveated_image,CV_8UC3);
    cv::resize(foveated_image,foveated_image,Size(size_map,size_map));

    return foveated_image;
}



//////////////////////////////
// FIXATION POINTS FUNTION  //
//////////////////////////////

std::vector<cv::Mat> FixationPoints (int img_size, int n_points, int random) {
    
    std::vector<cv::Mat> fixation_points;

    std::cout<<"Fixation Points: " << std::endl;
    for (int i = 0; i < n_points; i++) {
        cv::Mat fixation_point(2,1,CV_32S);
        if (random == 1){
            fixation_point.at<int>(0,0) = img_size*0.05 + rand() % (int)(img_size-img_size*0.05);
            fixation_point.at<int>(1,0) = img_size*0.05 + rand() % (int)(img_size-img_size*0.05);
        } 
        else {
            //fixation_point.at<int>(0,0) = img_size / sqrt(n_points) * sqrt(i) + (img_size / sqrt(n_points) /2);
            //fixation_point.at<int>(1,0) = img_size / sqrt(n_points) * sqrt(i) + (img_size/ sqrt(n_points) / 2);
        }
          
            //img_fov = foveate(img,img_size,levels,sigma,fixation_point);
            //img_fov = img_fov.clone();
            //cv::namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
            //imshow( "Display window", img_fov );
            std::cout<<"("<<fixation_point.at<int>(0,0)<<","<<fixation_point.at<int>(1,0)<< ')'<< std::endl;

            //waitKey(0);

            fixation_points.push_back(fixation_point);
    }        
    return fixation_points;
}

void VisualizeFixationPoints(int img_size, std::vector<cv::Mat> fixation_points) {

}
