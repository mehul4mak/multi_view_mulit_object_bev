#include <cmath>
#include <vector>
#include <opencv2/core.hpp>

int distance2(const cv::Point2f& a, const cv::Point2f& b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    return dx * dx + dy * dy;
}

std::vector<int> dbscan(const std::vector<cv::Point2f>& points, float eps, int minPts) {
    const int n = points.size();
    std::vector<bool> visited(n, false);
    std::vector<int> labels(n, -1); // -1 = noise
    int clusterId = 0;
    const float eps2 = eps * eps; // squared radius

    for (int i = 0; i < n; ++i) {
        if (visited[i])
            continue;

        visited[i] = true;

        std::vector<int> neighbors;
        for (int j = 0; j < n; ++j) {
            if (distance2(points[i], points[j]) <= eps2)
                neighbors.push_back(j);
        }

        if (neighbors.size() < minPts) {
            labels[i] = -1; // noise
            continue;
        }

        labels[i] = clusterId;
        for (size_t k = 0; k < neighbors.size(); ++k) {
            int j = neighbors[k];
            if (!visited[j]) {
                visited[j] = true;

                std::vector<int> neighbors2;
                for (int l = 0; l < n; ++l) {
                    if (distance2(points[j], points[l]) <= eps2)
                        neighbors2.push_back(l);
                }

                if (neighbors2.size() >= minPts) {
                    neighbors.insert(neighbors.end(), neighbors2.begin(), neighbors2.end());
                }
            }

            if (labels[j] == -1)
                labels[j] = clusterId;
        }

        ++clusterId;
    }

    return labels;
}
