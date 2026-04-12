#include "sam_client.hpp"

SAMClient::SAMClient(std::shared_ptr<rclcpp::Node> node) : nh_(node), enabled_(true), service_available_(false)
{
    client_node_ = std::make_shared<rclcpp::Node>("sam_service_client_isolated");
    sam_client_ = client_node_->create_client<vins::srv::SAMSegmentation>("sam_segmentation");
    
    // Wait for service to become available (with timeout)
    service_available_ = sam_client_->wait_for_service(std::chrono::seconds(30));
    
    if (service_available_)
    {
        RCLCPP_INFO(nh_->get_logger(), "SAM service connected successfully");
    }
    else
    {
        RCLCPP_WARN(nh_->get_logger(), "SAM service not available. Segmentation will be disabled.");
        enabled_ = false;
    }
}

SAMClient::~SAMClient()
{
}

bool SAMClient::isServiceAvailable()
{
    std::lock_guard<std::mutex> lock(mutex_);
    return service_available_ && enabled_;
}

bool SAMClient::checkServiceAvailability()
{
    std::lock_guard<std::mutex> lock(mutex_);
    service_available_ = sam_client_->service_is_ready();
    return service_available_;
}

bool SAMClient::getSegmentationMask(const cv::Mat& image, cv::Mat& mask)
{
    if (!enabled_ || !checkServiceAvailability())
    {
        return false;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    try
    {
        auto request = std::make_shared<vins::srv::SAMSegmentation::Request>();
        
        // Convert OpenCV image to ROS message
        cv_bridge::CvImage cv_image;
        cv_image.encoding = "bgr8";
        cv_image.image = image;
        cv_image.toImageMsg(request->image);
        
        // Call service
        auto result_future = sam_client_->async_send_request(request);
        
        if (rclcpp::spin_until_future_complete(client_node_, result_future, std::chrono::seconds(30)) ==
            rclcpp::FutureReturnCode::SUCCESS)
        {
            auto response = result_future.get();
            if (response->success)
            {
                // Convert ROS image message to OpenCV
                cv_bridge::CvImagePtr cv_ptr;
                try
                {
                    cv_ptr = cv_bridge::toCvCopy(response->mask, sensor_msgs::image_encodings::MONO8);
                    mask = cv_ptr->image.clone();
                    return true;
                }
                catch (cv_bridge::Exception& e)
                {
                    RCLCPP_ERROR(nh_->get_logger(), "cv_bridge exception: %s", e.what());
                    return false;
                }
            }
            else
            {
                RCLCPP_WARN(nh_->get_logger(), "SAM service returned failure");
                return false;
            }
        }
        else
        {
            RCLCPP_WARN(nh_->get_logger(), "Failed to call SAM service");
            service_available_ = false;
            return false;
        }
    }
    catch (const std::exception& e)
    {
        RCLCPP_ERROR(nh_->get_logger(), "Exception in getSegmentationMask: %s", e.what());
        return false;
    }
}

bool SAMClient::getSegmentationMaskWithPoints(const cv::Mat& image, 
                                              const std::vector<cv::Point2f>& points, 
                                              cv::Mat& mask)
{
    if (!enabled_ || !checkServiceAvailability())
    {
        return false;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    try
    {
        auto request = std::make_shared<vins::srv::SAMSegmentation::Request>();
        
        // Convert OpenCV image to ROS message
        cv_bridge::CvImage cv_image;
        cv_image.encoding = "bgr8";
        cv_image.image = image;
        cv_image.toImageMsg(request->image);
        
        // Add point prompts
        request->points.clear();
        for (const auto& pt : points)
        {
            geometry_msgs::msg::Point ros_pt;
            ros_pt.x = pt.x;
            ros_pt.y = pt.y;
            ros_pt.z = 0.0;
            request->points.push_back(ros_pt);
        }
        
        // Call service
        auto result_future = sam_client_->async_send_request(request);
        
        if (rclcpp::spin_until_future_complete(client_node_, result_future, std::chrono::seconds(30)) ==
            rclcpp::FutureReturnCode::SUCCESS)
        {
            auto response = result_future.get();
            if (response->success)
            {
                // Convert ROS image message to OpenCV
                cv_bridge::CvImagePtr cv_ptr;
                try
                {
                    cv_ptr = cv_bridge::toCvCopy(response->mask, sensor_msgs::image_encodings::MONO8);
                    mask = cv_ptr->image.clone();
                    return true;
                }
                catch (cv_bridge::Exception& e)
                {
                    RCLCPP_ERROR(nh_->get_logger(), "cv_bridge exception: %s", e.what());
                    return false;
                }
            }
            else
            {
                RCLCPP_WARN(nh_->get_logger(), "SAM service returned failure");
                return false;
            }
        }
        else
        {
            RCLCPP_WARN(nh_->get_logger(), "Failed to call SAM service");
            service_available_ = false;
            return false;
        }
    }
    catch (const std::exception& e)
    {
        RCLCPP_ERROR(nh_->get_logger(), "Exception in getSegmentationMaskWithPoints: %s", e.what());
        return false;
    }
}
