// types.h
#pragma once
#include <opencv2/core.hpp>
#include "nvdsmeta.h"

// Lightweight struct for storing object tracking info
// struct ObjectInfo {
//     NvDsObjectMeta* obj_meta_ptr = nullptr;
//     cv::Point2f bev_point;
//     int cluster_id = -1;
//     int global_id = -1;
// };


struct ObjectInfo {
    int global_id = -1;
    int object_id = -1;
    float confidence = -1.0;
    cv::Point2f bev_point;
    std::array<float, 256> reid_vector;
    int source_id = -1;
    int frame_num =-1;
    NvDsObjectMeta* obj_meta_ptr = nullptr;
};
