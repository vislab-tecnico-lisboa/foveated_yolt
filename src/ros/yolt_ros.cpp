#include "ros/yolt_ros.hpp"

YoltRos::YoltRos (const ros::NodeHandle & nh_) : nh(nh_), nh_priv("~")
{

		image_transport::ImageTransport it(nh);
		sub = it.subscribe("/usb_cam/image_raw", 1, &YoltRos::imageCallback, this);

		// Load network, pre-processment, set mean and load labels

		// Foveation parameters
		nh_priv.param<int>("width",   width,   500);
		nh_priv.param<int>("height",  height,  500);
		nh_priv.param<int>("levels",  levels,  10);
		nh_priv.param<int>("sigma_x", sigma_x, 70);
		nh_priv.param<int>("sigma_y", sigma_y, 70);

		string model_file;
		string weight_file;
		string mean_file;
		string label_file;
		string dataset_folder;

		nh_priv.param<std::string>("model_file", model_file, "");
		nh_priv.param<std::string>("weight_file", weight_file, "");
		nh_priv.param<std::string>("mean_file", mean_file, "");
		nh_priv.param<std::string>("label_file", label_file, "");

		ROS_INFO_STREAM("model_file: " << model_file);
		ROS_INFO_STREAM("weight_file: " << weight_file);
		ROS_INFO_STREAM("mean_file: " << mean_file);
		ROS_INFO_STREAM("label_file: " << label_file);

		fixation_point=cv::Mat(2,1,CV_32S);
		fixation_point.at<int>(0,0) = width/2;
		fixation_point.at<int>(1,0) = height/2;
		
		foveation=boost::shared_ptr<LaplacianBlending> (new LaplacianBlending(width, height, levels, sigma_x,sigma_y));

		conf_callback = boost::bind(&YoltRos::configCallback, this, _1, _2);
		server.setCallback(conf_callback);
		//network=boost::shared_ptr<Network>(new Network(model_file, weight_file, mean_file, label_file));
		//std::cout << "network initialized" << std::endl;
}

void YoltRos::configCallback(foveated_yolt::FoveaConfig &config, uint32_t level) {
	ROS_INFO_STREAM("Reconfigure Request: "<<  config.levels << " " << config.sigma_x << " " << config.sigma_y << " " << config.width << " " << config.height);

	this->sigma_x=config.sigma_x;
	this->sigma_y=config.sigma_y;

	if(config.levels!=this->levels || config.width!=this->width || config.height!=this->height)
	{
		foveation=boost::shared_ptr<LaplacianBlending> (new LaplacianBlending(this->width,this->height,config.levels,this->sigma_x,this->sigma_y));
		this->levels=config.levels;
		this->width =config.width;
		this->height=config.height;
	}
	else
	{
		foveation->CreateFilterPyr(this->width,this->height,this->sigma_x,this->sigma_y);
	}
}

void YoltRos::imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
	static int iteration=0;
	try
	{
		cv::Mat image;
		cv_bridge::toCvShare(msg, "bgr8")->image.convertTo(image, CV_64F);

		cv::resize(image,image,cv::Size(width,height));//resize image

		// Foveate
		cv::Mat foveated_image = foveation->Foveate(image,fixation_point);

		foveated_image.convertTo(foveated_image,CV_8UC3);

		cv::imshow("view",foveated_image);
		//std::cout << "iteration:" << ++iteration << std::endl;
		//std::cout << foveated_image.cols << std::endl;
		//std::cout << foveated_image.rows << std::endl;
	}
	catch (cv_bridge::Exception& e)
	{
		ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
	}
}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "image_listener");
	ros::NodeHandle nh;
	cv::namedWindow("view");
	cv::startWindowThread();

	YoltRos yolt_ros(nh);
	ros::spin();
	cv::destroyWindow("view");
}

