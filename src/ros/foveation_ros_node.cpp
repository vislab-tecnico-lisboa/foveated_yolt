#include "ros/foveation_ros.hpp"
int main(int argc, char **argv)
{
	ros::init(argc, argv, "image_listener");
	ros::NodeHandle nh;
	//cv::namedWindow("view");
	//cv::startWindowThread();
	std::string node_name="foveation";

	FoveationRos foveation_ros(nh,node_name);
	ros::spin();
	//cv::destroyWindow("view");
}
