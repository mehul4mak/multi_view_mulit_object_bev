// GlobalTracker.h
#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include "types.h" 

struct Track {
    int id;
    cv::Point2f last_point;
    int frames_since_seen;
};

class GlobalTracker {
    int next_id = 0;
    int max_age = 100;
    float match_thresh = 150.0;
    std::vector<Track> active_tracks;

public:
    void assignGlobalIDs(std::vector<ObjectInfo*>& objs);
    std::vector<int> assignGlobalIDsFromCentroids(const std::vector<cv::Point2f>& centroids);
};
