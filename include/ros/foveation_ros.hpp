#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <cv_bridge/cv_bridge.h>
#include "laplacian_foveation.hpp"
#include <memory>
#include <boost/shared_ptr.hpp>
#include <dynamic_reconfigure/server.h>
#include <foveated_yolt/FoveaConfig.h>
#include <foveated_yolt/EyeAction.h>
#include <actionlib/server/simple_action_server.h>

class FoveationRos
{
		int levels, width, height, sigma_x, sigma_y;

		boost::shared_ptr<LaplacianBlending> foveation;

		ros::NodeHandle nh, nh_priv;
		actionlib::SimpleActionServer<foveated_yolt::EyeAction> as_;
		std::string action_name_;
		image_transport::Subscriber sub;
		image_transport::Publisher pub;

		cv::Mat fixation_point;
		dynamic_reconfigure::Server<foveated_yolt::FoveaConfig> server;
		dynamic_reconfigure::Server<foveated_yolt::FoveaConfig>::CallbackType conf_callback;
		
		void configCallback(foveated_yolt::FoveaConfig &config, uint32_t level);


		foveated_yolt::EyeFeedback feedback_;
		foveated_yolt::EyeResult result_;
		void executeCB(const foveated_yolt::EyeGoalConstPtr &goal);

	public:
		FoveationRos (const ros::NodeHandle & nh_, const std::string & name);
		void imageCallback(const sensor_msgs::ImageConstPtr& msg);
};


