#include "ros/foveation_ros.hpp"
#include <chrono>
#include <ctime>

FoveationRos::FoveationRos (const ros::NodeHandle & nh_, const std::string & name) : nh(nh_), nh_priv("~"),
    as_(nh_, name, boost::bind(&FoveationRos::executeCB, this, _1), false),
    action_name_(name)
{
	image_transport::ImageTransport it(nh);
	std::cout << "network initialized" << std::endl;
	sub = it.subscribe("input_image", 5, &FoveationRos::imageCallback, this);
	pub = it.advertise("output_image", 5);
	// Load network, pre-processment, set mean and load labels

	// Foveation parameters
	nh_priv.param<int>("width",   width,   500);
	nh_priv.param<int>("height",  height,  500);
	nh_priv.param<int>("levels",  levels,  10);
	nh_priv.param<int>("sigma_x", sigma_x, 70);
	nh_priv.param<int>("sigma_y", sigma_y, 70);

	ROS_INFO_STREAM("width: " << width);
	ROS_INFO_STREAM("height: " << height);
	ROS_INFO_STREAM("levels: " << levels);
	ROS_INFO_STREAM("sigma_x: " << sigma_x);
	ROS_INFO_STREAM("sigma_y: " << sigma_y);

	fixation_point=cv::Mat(2,1,CV_32S);
	fixation_point.at<int>(0,0) = width/2;
	fixation_point.at<int>(1,0) = height/2;
	
	foveation=boost::shared_ptr<LaplacianBlending> (new LaplacianBlending(width, height, levels, sigma_x,sigma_y));

	conf_callback = boost::bind(&FoveationRos::configCallback, this, _1, _2);
	server.setCallback(conf_callback);
	//network=boost::shared_ptr<Network>(new Network(model_file, weight_file, mean_file, label_file));

	as_.start();
}

void FoveationRos::configCallback(foveated_yolt::FoveaConfig &config, uint32_t level) {
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

void FoveationRos::imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
	try
	{
		cv::Mat image;
		cv_bridge::toCvShare(msg, "bgr8")->image.convertTo(image, CV_64FC3);
		image=image/255.0;
		cv::resize(image,image,cv::Size(this->width,this->height));//resize image

		// Foveate
		cv::Mat foveated_image = foveation->Foveate(image,fixation_point);

		// Visualize

		//cv::Point center_(fixation_point.at<int>(0,0),fixation_point.at<int>(1,0));
		//cv::Scalar color_=cv::Scalar(255,0,0);
		//cv::circle(foveated_image, center_ , 5, color_);
		//RotatedRect (const Point2f &center, const Size2f &size, float angle)
		//cv::ellipse(foveated_image, const RotatedRect& box, color_);
		//std::cout << foveated_image << std::endl;
		//foveated_image.convertTo(foveated_image,CV_8UC3);
    		//cv::cvtColor(top_final_saliency_maps[0], top_final_saliency_maps[0], cv::COLOR_GRAY2BGR);
		foveated_image.convertTo(foveated_image,CV_8UC3, 255.0);
		sensor_msgs::ImagePtr msg_out = cv_bridge::CvImage(std_msgs::Header(), "bgr8", foveated_image).toImageMsg();

		pub.publish(msg_out);
	}
	catch (cv_bridge::Exception& e)
	{
		ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
	}
	std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();
	ROS_DEBUG_STREAM("total yolt time: " << duration << " ms");
}


void FoveationRos::executeCB(const foveated_yolt::EyeGoalConstPtr &goal)
{
	// helper variables
	bool success = true;

	this->sigma_x=goal->sigma_xx;
	this->sigma_y=goal->sigma_yy;

	if(goal->levels!=this->levels)
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
	}

}


