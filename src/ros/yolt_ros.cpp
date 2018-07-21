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
	string dataset_folder;

	nh_priv.param<int>("top_classes", top_classes, 1);
	nh_priv.param<std::string>("model_file", model_file, "");
	nh_priv.param<std::string>("weight_file", weight_file, "");
	nh_priv.param<std::string>("mean_file", mean_file, "");
	nh_priv.param<std::string>("label_file", label_file, "");

	ROS_INFO_STREAM("model_file: " << model_file);
	ROS_INFO_STREAM("weight_file: " << weight_file);
	ROS_INFO_STREAM("mean_file: " << mean_file);
	ROS_INFO_STREAM("label_file: " << label_file);

	Caffe::set_mode(Caffe::GPU);
	Caffe::SetDevice(0);
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
		cv_bridge::toCvShare(msg, "bgr8")->image.convertTo(image, CV_64F);

		//cv::resize(image,image,cv::Size(width,height));//resize image

		ClassData first_pass_data = yolt_network->Classify(image, top_classes);
		std::vector<Rect> bounding_boxes;
		std::vector<std::string> labels;
		std::vector<float> scores;
		std::vector<int> indexes;
		std::vector<cv::Mat> saliency_maps;
		float thresh=0.7;

		// Weak object localization
		for (unsigned int class_index = 0; class_index < top_classes; ++class_index) {
			cv::Mat saliency_map;
			cv::Rect bounding_box = yolt_network->CalcBBox(class_index,image, first_pass_data, (float)thresh,saliency_map);



			// Save all labels, scores, indexes, bounding boxes and saliency maps
			bounding_boxes.push_back(bounding_box);
			saliency_maps.push_back(saliency_map);
			labels.push_back(first_pass_data.label[class_index]);
			scores.push_back(first_pass_data.score[class_index]);
			indexes.push_back(first_pass_data.index[class_index]);

		}

		// Task free: Get top class
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
			}
			++top;
		}
		//std::cout << saliency_maps[0] << std::endl;
		//std::cout << top_final_index[0] << ": " << top_final_labels[0] << " " << top_final_scores[0] << std::endl;
		saliency_maps[0]=255.0*saliency_maps[0];
    		saliency_maps[0].convertTo(saliency_maps[0],CV_8UC1); 
    		cv::cvtColor(saliency_maps[0], saliency_maps[0], cv::COLOR_GRAY2BGR);

		sensor_msgs::ImagePtr msg_out = cv_bridge::CvImage(std_msgs::Header(), "bgr8", saliency_maps[0]).toImageMsg();

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

