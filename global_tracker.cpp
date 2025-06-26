#include <iostream>
#include <opencv2/core.hpp>
#include "global_tracker.h"

// Assign global IDs to incoming objects using nearest-neighbor tracking
void GlobalTracker::assignGlobalIDs(std::vector<ObjectInfo*>& objs) {
    std::cout << "[Tracker] Received " << objs.size() << " objects for tracking.\n";

    if (objs.empty()) {
        std::cout << "[Tracker] No objects to process. Exiting.\n";
        return;
    }

    std::vector<bool> matched(objs.size(), false);
    std::vector<bool> track_matched(active_tracks.size(), false);

    // 🔍 Step 1: Try matching each object to existing tracks
    for (size_t i = 0; i < objs.size(); ++i) {
        if (!objs[i]) {
            std::cerr << "[Tracker][ERROR] Null pointer found in objs at index " << i << ". Skipping.\n";
            continue;
        }

        float min_dist = match_thresh;
        int best_track_idx = -1;

        for (size_t j = 0; j < active_tracks.size(); ++j) {
            float dist = cv::norm(objs[i]->bev_point - active_tracks[j].last_point);
            if (dist < min_dist) {
                min_dist = dist;
                best_track_idx = j;
            }
        }

        if (best_track_idx >= 0) {
            std::cout << "[Tracker] Matched object " << i << " to track ID "
                      << active_tracks[best_track_idx].id << "\n";

            objs[i]->global_id = active_tracks[best_track_idx].id;
            active_tracks[best_track_idx].last_point = objs[i]->bev_point;
            active_tracks[best_track_idx].frames_since_seen = 0;
            matched[i] = true;

            // Resize if new track was added previously
            if (best_track_idx >= static_cast<int>(track_matched.size())) {
                track_matched.resize(best_track_idx + 1, false);
            }
            track_matched[best_track_idx] = true;
        } else {
            std::cout << "[Tracker] No match for object " << i << ", will assign new ID.\n";
        }
    }

    // 🆕 Step 2: Assign new IDs to unmatched objects
    for (size_t i = 0; i < objs.size(); ++i) {
        if (!objs[i]) continue;

        if (!matched[i]) {
            objs[i]->global_id = next_id++;
            std::cout << "[DEBUG] Creating track: ID=" << objs[i]->global_id
                      << ", BEV=(" << objs[i]->bev_point.x << "," << objs[i]->bev_point.y << "), age=0\n";

            active_tracks.push_back({objs[i]->global_id, objs[i]->bev_point, 0});
            std::cout << "[Tracker] Assigned NEW track ID " << objs[i]->global_id << " to object " << i << "\n";
        }
    }

    // 🧹 Step 3: Remove old (aged) tracks
    std::vector<Track> new_tracks;
    std::cout << "[Tracker] Starting cleanup. Active tracks before: " << active_tracks.size()
              << ", track_matched size: " << track_matched.size() << "\n";

    for (size_t j = 0; j < active_tracks.size(); ++j) {
        bool matched_this_track = (j < track_matched.size()) ? track_matched[j] : false;

        if (!matched_this_track)
            active_tracks[j].frames_since_seen++;

        if (active_tracks[j].frames_since_seen < max_age) {
            new_tracks.push_back(active_tracks[j]);
        } else {
            std::cout << "[Tracker] Removing aged-out track ID " << active_tracks[j].id << "\n";
        }
    }

    active_tracks = std::move(new_tracks);

    std::cout << "[Tracker] Tracking complete. Active tracks: " << active_tracks.size() << "\n";
}

std::vector<int> GlobalTracker::assignGlobalIDsFromCentroids(const std::vector<cv::Point2f>& centroids) {
    std::vector<int> assigned_ids;
    std::vector<bool> track_matched(active_tracks.size(), false);

    std::cout << "[Tracker] Received " << centroids.size() << " centroids for tracking.\n";

    for (size_t i = 0; i < centroids.size(); ++i) {
        const auto& pt = centroids[i];
        float min_dist = match_thresh;
        int best_track_idx = -1;

        // Find the best matching track within threshold
        for (size_t j = 0; j < active_tracks.size(); ++j) {
            float dist = cv::norm(pt - active_tracks[j].last_point);
            if (dist < min_dist) {
                min_dist = dist;
                best_track_idx = j;
            }
        }

        if (best_track_idx >= 0) {
            // Match found
            int gid = active_tracks[best_track_idx].id;
            assigned_ids.push_back(gid);
            active_tracks[best_track_idx].last_point = pt;
            active_tracks[best_track_idx].frames_since_seen = 0;
            track_matched[best_track_idx] = true;

            std::cout << "[Tracker] Matched centroid " << i << " with track ID " << gid << "\n";
        } else {
            // No match → create new track
            int new_id = next_id++;
            active_tracks.push_back({new_id, pt, 0});
            track_matched.push_back(true);
            assigned_ids.push_back(new_id);

            std::cout << "[Tracker] Assigned NEW track ID " << new_id << " to centroid " << i << "\n";
        }
    }

    // Age and prune old tracks
    std::vector<Track> new_tracks;
    for (size_t j = 0; j < active_tracks.size(); ++j) {
        if (!track_matched[j])
            active_tracks[j].frames_since_seen++;
        if (active_tracks[j].frames_since_seen < max_age)
            new_tracks.push_back(active_tracks[j]);
        else
            std::cout << "[Tracker] Pruned old track ID " << active_tracks[j].id << "\n";
    }
    active_tracks = std::move(new_tracks);

    return assigned_ids;
}
