#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <cv_bridge/cv_bridge.h>
#include "laplacian_foveation.hpp"
#include "network_classes.hpp"
#include <memory>
#include <boost/shared_ptr.hpp>
#include <dynamic_reconfigure/server.h>
#include <foveated_yolt/FoveaConfig.h>

class YoltRos
{
		int levels, width, height, sigma_x, sigma_y;

		boost::shared_ptr<LaplacianBlending> foveation;
		boost::shared_ptr<Network> network;

		ros::NodeHandle nh, nh_priv;
		image_transport::Subscriber sub;

		cv::Mat fixation_point;
		dynamic_reconfigure::Server<foveated_yolt::FoveaConfig> server;
		dynamic_reconfigure::Server<foveated_yolt::FoveaConfig>::CallbackType conf_callback;
		
		void configCallback(foveated_yolt::FoveaConfig &config, uint32_t level);
	public:
		YoltRos (const ros::NodeHandle & nh_);
		void imageCallback(const sensor_msgs::ImageConstPtr& msg);
};


