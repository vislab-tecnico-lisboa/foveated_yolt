#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <cv_bridge/cv_bridge.h>
#include "network_classes.hpp"
#include <memory>
#include <boost/shared_ptr.hpp>
#include <dynamic_reconfigure/server.h>
#include <foveated_yolt/FoveaConfig.h>
#include <foveated_yolt/TaskAction.h>
#include <actionlib/server/simple_action_server.h>

class YoltRos
{
		double saliency_threshold;
		int top_classes, width, height;

		boost::shared_ptr<Network> yolt_network;

		ros::NodeHandle nh, nh_priv;
		actionlib::SimpleActionServer<foveated_yolt::TaskAction> as_;
		std::string action_name_;
		image_transport::Subscriber sub;
		image_transport::Publisher pub;

		cv::Mat fixation_point;
		dynamic_reconfigure::Server<foveated_yolt::FoveaConfig> server;
		dynamic_reconfigure::Server<foveated_yolt::FoveaConfig>::CallbackType conf_callback;
		


		foveated_yolt::TaskFeedback feedback_;
		foveated_yolt::TaskResult result_;
		void executeCB(const foveated_yolt::TaskGoalConstPtr &goal);

	public:
		YoltRos (const ros::NodeHandle & nh_, const std::string & name);
		void imageCallback(const sensor_msgs::ImageConstPtr& msg);
};


