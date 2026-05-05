/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#pragma once

// #define GPU_MODE 1


#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>
#include <thread>
#include <mutex>
#include <atomic>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

#ifdef GPU_MODE
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#endif

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "../estimator/parameters.h"
#include "../sam_service/sam_client.hpp"
#include "../utility/tic_toc.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;


#define ROS_INFO RCUTILS_LOG_INFO
#define ROS_WARN RCUTILS_LOG_WARN
#define ROS_DEBUG RCUTILS_LOG_DEBUG
#define ROS_ERROR RCUTILS_LOG_ERROR

bool inBorder(const cv::Point2f &pt);
void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
void reduceVector(vector<int> &v, vector<uchar> status);

class FeatureTracker
{
public:
    FeatureTracker();
    ~FeatureTracker() {
        if (sam_thread_.joinable()) {
            sam_thread_.join();
        }
    }
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> trackImage(double _cur_time, const cv::Mat &_img, const cv::Mat &_img1 = cv::Mat());
    void setMask();
    void addPoints();
    void readIntrinsicParameter(const vector<string> &calib_file);
    void showUndistortion(const string &name);
    void rejectWithF();
    void undistortedPoints();
    vector<cv::Point2f> undistortedPts(vector<cv::Point2f> &pts, camodocal::CameraPtr cam);
    vector<cv::Point2f> ptsVelocity(vector<int> &ids, vector<cv::Point2f> &pts, 
                                    map<int, cv::Point2f> &cur_id_pts, map<int, cv::Point2f> &prev_id_pts);
    void showTwoImage(const cv::Mat &img1, const cv::Mat &img2, 
                      vector<cv::Point2f> pts1, vector<cv::Point2f> pts2);
    void drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight, 
                                   vector<int> &curLeftIds,
                                   vector<cv::Point2f> &curLeftPts, 
                                   vector<cv::Point2f> &curRightPts,
                                   map<int, cv::Point2f> &prevLeftPtsMap);
    void setPrediction(map<int, Eigen::Vector3d> &predictPts);
    double distance(cv::Point2f &pt1, cv::Point2f &pt2);
    void removeOutliers(set<int> &removePtsIds);
    cv::Mat getTrackImage();
    bool inBorder(const cv::Point2f &pt);
    vector<cv::Point2f> getTrackedPts();

    // SAM Integration
    void initSAM(bool use_sam, double update_interval = 5.0);
    void setSAMClient(std::shared_ptr<SAMClient> client);
    cv::Mat getMask();

    int row, col;
    cv::Mat imTrack;
    cv::Mat mask;
    cv::Mat fisheye_mask;
    cv::Mat prev_img, cur_img;
    vector<cv::Point2f> n_pts;
    vector<cv::Point2f> predict_pts;
    vector<cv::Point2f> predict_pts_debug;
    vector<cv::Point2f> prev_pts, cur_pts, cur_right_pts;
    vector<cv::Point2f> prev_un_pts, cur_un_pts, cur_un_right_pts;
    vector<cv::Point2f> pts_velocity, right_pts_velocity;
    vector<int> ids, ids_right;
    vector<int> track_cnt;

    // SAM Integration
    std::shared_ptr<SAMClient> sam_client_;
    bool use_sam_;
    double sam_update_interval_;
    int frame_count_;
    cv::Mat sam_mask;
    
    std::mutex sam_mutex_;
    std::atomic<bool> sam_processing_;
    std::thread sam_thread_;
    std::atomic<double> last_sam_time_;
    
    // Intelligent Trigger variables
    std::set<int> last_sam_feature_ids_;
    std::atomic<double> current_angular_vel_{0.0};   // norm, used by blur gate
    std::atomic<double> current_angular_vel_x_{0.0};
    std::atomic<double> current_angular_vel_y_{0.0};
    std::atomic<double> current_angular_vel_z_{0.0};
    std::atomic<double> current_imu_dt_{0.0};
    std::atomic<double> current_imu_cov_trace_{0.0};
    std::atomic<bool> force_sam_trigger_{false};
    std::atomic<bool> sam_pose_sync_needed_{false};

    void samThreadMethod(cv::Mat image);
    void updateIMUKinematics(const Eigen::Vector3d &angular_vel_vec, double dt, double cov_trace);

    std::atomic<int> sam_invoked_{0};
    std::atomic<double> sam_start_time_log_{0.0};
    std::atomic<double> sam_end_time_log_{0.0};
    std::atomic<double> sam_duration_log_{0.0};
    std::atomic<int> gate_blocked_{0};
    std::atomic<double> mask_iou_{0.0};

    map<int, cv::Point2f> cur_un_pts_map, prev_un_pts_map;
    map<int, cv::Point2f> cur_un_right_pts_map, prev_un_right_pts_map;
    map<int, cv::Point2f> prevLeftPtsMap;
    vector<camodocal::CameraPtr> m_camera;
    double cur_time;
    double prev_time;
    bool stereo_cam;
    int n_id;
    bool hasPrediction;
};
