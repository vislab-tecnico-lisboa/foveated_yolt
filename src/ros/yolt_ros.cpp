#include "ros/yolt_ros.hpp"

YoltRos::YoltRos (const ros::NodeHandle & nh_, const std::string & name) : nh(nh_), nh_priv("~"),
    as_(nh_, name, boost::bind(&YoltRos::executeCB, this, _1), false),
    action_name_(name)
{
	image_transport::ImageTransport it(nh);

	sub = it.subscribe("/input_image", 1, &YoltRos::imageCallback, this);
	pub = it.advertise("/output", 1);
	// Load network, pre-processment, set mean and load labels

	// Network parameters
	string model_file;
	string weight_file;
	string mean_file;
	string label_file;
	string device;
	int device_id;
	nh_priv.param<int>("top_classes", top_classes, 5);
	nh_priv.param<int>("width", width, 227);
	nh_priv.param<int>("height", height, 227);
	nh_priv.param<std::string>("model_file", model_file, "");
	nh_priv.param<std::string>("weight_file", weight_file, "");
	nh_priv.param<std::string>("mean_file", mean_file, "");
	nh_priv.param<std::string>("label_file", label_file, "");
	nh_priv.param<std::string>("device", device, "CPU");

	nh_priv.param<int>("device_id", device_id, 0);
	ROS_INFO_STREAM("model_file: " << model_file);
	ROS_INFO_STREAM("weight_file: " << weight_file);
	ROS_INFO_STREAM("mean_file: " << mean_file);
	ROS_INFO_STREAM("label_file: " << label_file);
	ROS_INFO_STREAM("device: " << device);
	ROS_INFO_STREAM("device_id: " << device_id);
	ROS_INFO_STREAM("top_classes: " << top_classes);

	if (device=="CPU")                           // Set Mode
		Caffe::set_mode(Caffe::CPU);
	else {
		Caffe::set_mode(Caffe::GPU);
		Caffe::SetDevice(device_id);
	}

	yolt_network=boost::shared_ptr<Network>(new Network(model_file, weight_file, mean_file, label_file));

	//conf_callback = boost::bind(&YoltRos::configCallback, this, _1, _2);
	//server.setCallback(conf_callback);

	std::cout << "network initialized" << std::endl;
	as_.start();
}


void YoltRos::imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
	try
	{
		cv::Mat image;
		cv_bridge::toCvShare(msg, "bgr8")->image.convertTo(image, CV_8UC3);
		cv::resize(image,image,cv::Size(width,height));//resize image
		//image.convertTo(image, CV_32FC3);
		//image=image/255.0;
		ClassData first_pass_data = yolt_network->Classify(image, top_classes);

		std::vector<Rect> bounding_boxes;
		std::vector<std::string> labels;
		std::vector<float> scores;
		std::vector<int> indexes;
		std::vector<cv::Mat> saliency_maps;


		//std::cout << image << std::endl;
	    	//cv::normalize(image, image); 
		//std::cout << image << std::endl;
		//std::cout << image << std::endl;




		float thresh=0.5;

		// Weak object localization
		for (unsigned int class_index = 0; class_index < top_classes; ++class_index) {
			cv::Mat saliency_map;
			cv::Rect bounding_box = yolt_network->CalcBBox(class_index, first_pass_data, (float)thresh, saliency_map);
			//std::cout << bounding_box << std::endl;
			// Save all labels, scores, indexes, bounding boxes and saliency maps
			for (unsigned int class_index_bb = 0; class_index_bb < top_classes; ++class_index_bb) {
				bounding_boxes.push_back(bounding_box);
				saliency_maps.push_back(saliency_map);
				labels.push_back(first_pass_data.label[class_index_bb]);
				scores.push_back(first_pass_data.score[class_index_bb]);
				indexes.push_back(first_pass_data.index[class_index_bb]);
			}
		}

		// Task free: Get top class
		std::vector<cv::Mat> top_final_saliency_maps;
		std::vector<std::string> top_final_labels;
		std::vector<float> top_final_scores;
		std::vector<int> top_final_index;
		std::vector<int> sort_scores_index  = ArgMax(scores, top_classes*top_classes);
		int top = 0;

		while(top_final_labels.size() < top_classes){
			int idx = sort_scores_index[top];
			if((std::find(top_final_labels.begin(), top_final_labels.end(), labels[idx])) == (top_final_labels.end())) {
				top_final_labels.push_back(labels[idx]);
				top_final_scores.push_back(scores[idx]);
				top_final_index.push_back(indexes[idx]);    
				top_final_saliency_maps.push_back(saliency_maps[idx]);         
			}
			++top;
		}

		for (unsigned int class_index = 0; class_index < top_classes; ++class_index) {

			std::cout << top_final_index[class_index] << ": " << top_final_labels[class_index] << " " << top_final_scores[class_index] << " :::: ";
		}
		std::cout << std::endl;



		//top_final_saliency_maps[0]=255.0*top_final_saliency_maps[0];

    		//cv::cvtColor(top_final_saliency_maps[0], top_final_saliency_maps[0], cv::COLOR_GRAY2BGR);
    		//top_final_saliency_maps[0].convertTo(top_final_saliency_maps[0],CV_8UC1); 

		sensor_msgs::ImagePtr msg_out = cv_bridge::CvImage(std_msgs::Header(), "bgr8", top_final_saliency_maps[0]).toImageMsg();

		pub.publish(msg_out);
	}
	catch (cv_bridge::Exception& e)
	{
		ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
	}
}


void YoltRos::executeCB(const foveated_yolt::TaskGoalConstPtr &goal)
{
	// helper variables
	bool success = true;


	/*if(goal->levels!=this->levels)
	{
		foveation=boost::shared_ptr<LaplacianBlending> (new LaplacianBlending(this->width,this->height,goal->levels,this->sigma_x,this->sigma_y));
		this->levels=goal->levels;
	}
	else
	{
		foveation->CreateFilterPyr(this->width,this->height,this->sigma_x,this->sigma_y);
	}

	fixation_point.at<int>(0,0)=goal->cx;  fixation_point.at<int>(1,0)=goal->cy;
	// start executing the action

	if(success)
	{
		result_.sequence = feedback_.sequence;
		ROS_INFO("%s: Succeeded", action_name_.c_str());
		// set the action state to succeeded
		as_.setSucceeded(result_);
	}*/

}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "image_listener");
	ros::NodeHandle nh;
	//cv::namedWindow("view");
	//cv::startWindowThread();
	std::string node_name="foveation";

	YoltRos yolt_ros(nh,node_name);
	ros::spin();
	//cv::destroyWindow("view");
}

