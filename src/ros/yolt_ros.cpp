#include "ros/yolt_ros.hpp"

YoltRos::YoltRos (const ros::NodeHandle & nh_, const int & _width, const int & _height, const int & levels_, const int & sigma_x_, const int & sigma_y_, const int & size_map_) : nh(nh_), levels(levels_), sigma_x(sigma_x_), sigma_y(sigma_y_), size_map(size_map_) 
{
		foveation=boost::shared_ptr<LaplacianBlending> (new LaplacianBlending(_width, _height, levels, sigma_x,sigma_y));
		image_transport::ImageTransport it(nh);
		sub = it.subscribe("/usb_cam/image_raw", 1, &YoltRos::imageCallback, this);

		// Load network, pre-processment, set mean and load labels
		string model_file;
		string weight_file;
		string mean_file;
		string label_file;
		string dataset_folder;

		nh_.param<std::string>("model_file", model_file, "");
		nh_.param<std::string>("weight_file", weight_file, "");
		nh_.param<std::string>("mean_file", mean_file, "");
		nh_.param<std::string>("label_file", label_file, "");

		network=boost::shared_ptr<Network>(new Network(model_file, weight_file, mean_file, label_file));
		std::cout << "network initialized" << std::endl;
}


void YoltRos::imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
	try
	{
		cv::Mat image;
		cv_bridge::toCvShare(msg, "bgr8")->image.convertTo(image, CV_64F);

		cv::Mat fixation_point(2,1,CV_32S);
		fixation_point.at<int>(0,0) = image.rows/2;
		fixation_point.at<int>(1,0) = image.cols/2;

		// Foveate
		cv::Mat foveated_image = foveation->Foveate(image,fixation_point);

		foveated_image.convertTo(foveated_image,CV_8UC3);
		cv::resize(foveated_image,foveated_image,Size(size_map,size_map));

		cv::imshow("view",foveated_image);
		//cv::waitKey(30);
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

	int width=640; int height=480; int levels=5; int sigma=70; int size_map=227;


	YoltRos yolt_ros(nh,width,height,levels,sigma,sigma,size_map);
	ros::spin();
	cv::destroyWindow("view");
}

