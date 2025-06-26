// cluster_meta.h
#pragma once
#include <vector>
#include <gst/gst.h>
#include <opencv2/core.hpp>

struct ClusterMeta {
    std::vector<cv::Point2f> centroids;
    int frame_num;
    int source_id;
};

