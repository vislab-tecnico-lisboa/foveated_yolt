#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <cv_bridge/cv_bridge.h>
#include "laplacian_foveation.hpp"
#include "network_classes.hpp"
#include <memory>
#include <boost/shared_ptr.hpp>

class YoltRos
{
		int levels;
		int sigma_x,sigma_y;
		int size_map;
		boost::shared_ptr<LaplacianBlending> foveation;
		ros::NodeHandle nh;
		image_transport::Subscriber sub;
		boost::shared_ptr<Network> network;
	public:
		YoltRos (const ros::NodeHandle & nh_, const int & _width, const int & _height, const int & levels_, const int & sigma_x_, const int & sigma_y_, const int & size_map_);
		void imageCallback(const sensor_msgs::ImageConstPtr& msg);
};


