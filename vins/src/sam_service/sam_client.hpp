#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <vins/srv/sam_segmentation.hpp>
#include <mutex>
#include <memory>
#include <vector>

class SAMClient
{
public:
    SAMClient(std::shared_ptr<rclcpp::Node> node);
    ~SAMClient();
    
    /**
     * Get segmentation mask for an image
     * @param image Input image (BGR format)
     * @param mask Output binary mask (255 for segmented regions, 0 otherwise)
     * @return true if successful, false otherwise
     */
    bool getSegmentationMask(const cv::Mat& image, cv::Mat& mask);
    
    /**
     * Get segmentation mask with point prompts
     * @param image Input image (BGR format)
     * @param points Point prompts for segmentation
     * @param mask Output binary mask
     * @return true if successful, false otherwise
     */
    bool getSegmentationMaskWithPoints(const cv::Mat& image, 
                                       const std::vector<cv::Point2f>& points, 
                                       cv::Mat& mask);
    
    /**
     * Check if SAM service is available
     * @return true if service is available
     */
    bool isServiceAvailable();
    
    /**
     * Enable/disable SAM integration
     */
    void setEnabled(bool enabled) { enabled_ = enabled; }
    bool isEnabled() const { return enabled_; }

private:
    std::shared_ptr<rclcpp::Node> nh_;
    rclcpp::Client<vins::srv::SAMSegmentation>::SharedPtr sam_client_;
    bool enabled_;
    std::mutex mutex_;
    bool service_available_;
    
    bool checkServiceAvailability();
};
