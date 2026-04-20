/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "utility.h"

Eigen::Matrix3d Utility::g2R(const Eigen::Vector3d &g)
{
    Eigen::Matrix3d R0;
    Eigen::Vector3d ng1 = g.normalized();
    Eigen::Vector3d ng2{0, 0, 1.0};
    R0 = Eigen::Quaterniond::FromTwoVectors(ng1, ng2).toRotationMatrix();
    double yaw = Utility::R2ypr(R0).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    // R0 = Utility::ypr2R(Eigen::Vector3d{-90, 0, 0}) * R0;
    return R0;
}

#include <fstream>
#include <iostream>

double Utility::getCpuUsage() {
    static unsigned long long lastTotalUser = 0, lastTotalUserLow = 0, lastTotalSys = 0, lastTotalIdle = 0;
    FILE* file = fopen("/proc/stat", "r");
    if (!file) return -1.0;
    unsigned long long totalUser = 0, totalUserLow = 0, totalSys = 0, totalIdle = 0;
    if (fscanf(file, "cpu %llu %llu %llu %llu", &totalUser, &totalUserLow, &totalSys, &totalIdle) != 4) {
        fclose(file);
        return -1.0;
    }
    fclose(file);
    
    double percent = 0.0;
    if (lastTotalUser != 0) {
        unsigned long long totalDiff = (totalUser - lastTotalUser) + (totalUserLow - lastTotalUserLow) + (totalSys - lastTotalSys);
        unsigned long long idleDiff = totalIdle - lastTotalIdle;
        if (totalDiff + idleDiff > 0) percent = (double)totalDiff / (double)(totalDiff + idleDiff) * 100.0;
    }
    
    lastTotalUser = totalUser;
    lastTotalUserLow = totalUserLow;
    lastTotalSys = totalSys;
    lastTotalIdle = totalIdle;
    return percent;
}

double Utility::getGpuUsage() {
    std::ifstream file("/sys/devices/gpu.0/load"); // Tegra/Jetson
    if (!file.is_open()) return 0.0;
    double load = 0.0;
    if (file >> load) return load / 10.0; // Jetson load format (per mille)
    return 0.0;
}

