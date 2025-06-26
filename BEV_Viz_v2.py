import cv2
import numpy as np
from kafka import KafkaConsumer
import json
from collections import defaultdict
# Define the Kafka topic and broker address
KAFKA_BROKER = 'localhost:9092'  # Replace with your broker address if needed
TOPIC_NAME = 't2'

_bev = cv2.imread("peopletrack/bev.jpg")
output_bev_size = _bev.shape[1::-1]
output_bev_size

import numpy as np
from scipy.spatial.distance import euclidean
from collections import deque, defaultdict

class SimpleTrack:
    def __init__(self, track_id, position):
        self.track_id = track_id
        self.positions = deque(maxlen=100)
        self.positions.append(position)
        self.missing_frames = 0
        self.age = 1  # number of frames alive
        self.confirmed = False

    def update(self, position):
        self.positions.append(position)
        self.missing_frames = 0
        self.age += 1
        if self.age >= 3:  # confirm only if seen 3+ times
            self.confirmed = True

    def predict(self):
        if len(self.positions) >= 2:
            dx = self.positions[-1][0] - self.positions[-2][0]
            dy = self.positions[-1][1] - self.positions[-2][1]
            pred = (self.positions[-1][0] + dx, self.positions[-1][1] + dy)
            return pred
        return self.positions[-1]

class CentroidTracker:
    def __init__(self, max_distance=40, max_missing=5):
        self.tracks = {}
        self.next_id = 0
        self.max_distance = max_distance
        self.max_missing = max_missing

    def update(self, detections):
        # Step 1: match detections to existing tracks
        assignments = {}
        used_tracks = set()
        used_detections = set()

        for tid, track in self.tracks.items():
            min_dist = float('inf')
            matched = None
            for i, det in enumerate(detections):
                if i in used_detections:
                    continue
                dist = euclidean(track.positions[-1], det)
                if dist < min_dist and dist < self.max_distance:
                    min_dist = dist
                    matched = i
            if matched is not None:
                track.update(detections[matched])
                assignments[tid] = detections[matched]
                used_tracks.add(tid)
                used_detections.add(matched)

        # Step 2: unmatched detections → new tracks
        for i, det in enumerate(detections):
            if i not in used_detections:
                new_track = SimpleTrack(self.next_id, det)
                self.tracks[self.next_id] = new_track
                self.next_id += 1

        # Step 3: unmatched tracks → increase missing count
        for tid, track in self.tracks.items():
            if tid not in used_tracks:
                track.missing_frames += 1

        # Step 4: delete lost tracks
        self.tracks = {tid: trk for tid, trk in self.tracks.items()
                       if trk.missing_frames <= self.max_missing}

        return self.get_confirmed_tracks()

    def get_confirmed_tracks(self):
        return {tid: trk.positions[-1] for tid, trk in self.tracks.items() if trk.confirmed}

import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import deque

class KalmanTrack:
    def __init__(self, track_id, initial_point):
        self.track_id = track_id
        self.kf = self._init_kalman_filter(initial_point)
        self.positions = deque(maxlen=100)
        self.positions.append(initial_point)
        self.age = 1
        self.hits = 1
        self.confirmed = False
        self.time_since_update = 0
        self.last_prediction = initial_point

    def _init_kalman_filter(self, pt):
        kf = {
            'state': np.array([pt[0], pt[1], 0, 0], dtype=float),
            'P': np.eye(4) * 500,
            'F': np.array([[1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]]),
            'H': np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]]),
            'R': np.eye(2) * 2.0,
            'Q': np.eye(4) * 0.1
        }
        return kf

    def predict(self):
        F, Q = self.kf['F'], self.kf['Q']
        self.kf['state'] = F @ self.kf['state']
        self.kf['P'] = F @ self.kf['P'] @ F.T + Q
        self.last_prediction = tuple(self.kf['state'][:2])
        return self.last_prediction

    def update(self, pt, damping_factor=0.6):
        z = np.array(pt)
        H, R = self.kf['H'], self.kf['R']
        y = z - H @ self.kf['state']
        S = H @ self.kf['P'] @ H.T + R
        K = self.kf['P'] @ H.T @ np.linalg.inv(S)

        raw_state_update = self.kf['state'] + K @ y

        # 🔧 Damping: blend prediction and update to reduce jitter
        self.kf['state'] = (
            damping_factor * raw_state_update +
            (1 - damping_factor) * self.kf['state']
        )

        self.kf['P'] = (np.eye(4) - K @ H) @ self.kf['P']

        # Save smoothed position
        self.positions.append(tuple(self.kf['state'][:2]))
        self.age += 1
        self.hits += 1
        self.time_since_update = 0
        if self.hits >= 10:
            self.confirmed = True


    def get_prediction(self):
        return self.last_prediction


from scipy.optimize import linear_sum_assignment

class KalmanTrackerManager:
    def __init__(self, max_age=5, match_threshold=30, min_hits=3):
        self.tracks = {}
        self.next_id = 0
        self.max_age = max_age
        self.match_threshold = match_threshold
        self.min_hits = min_hits

    def _cost_matrix(self, detections):
        cost = []
        for track in self.tracks.values():
            pred = track.predict()
            cost.append([np.linalg.norm(np.array(pred) - np.array(d)) for d in detections])
        return np.array(cost)

    def update(self, detections):
        detections = [tuple(d) for d in detections]

        if not self.tracks:
            for d in detections:
                self._start_new_track(d)
            return self._get_confirmed_tracks()

        cost = self._cost_matrix(detections)
        row_idx, col_idx = linear_sum_assignment(cost)

        assigned_dets = set()
        assigned_trks = set()
        track_keys = list(self.tracks.keys())

        # ✅ Step 1: Assign detections to tracks
        for r, c in zip(row_idx, col_idx):
            if cost[r][c] < self.match_threshold:
                tid = track_keys[r]
                self.tracks[tid].update(detections[c])
                assigned_trks.add(tid)
                assigned_dets.add(c)

        # ✅ Step 2: Increase age of unmatched tracks
        for tid in self.tracks:
            if tid not in assigned_trks:
                self.tracks[tid].time_since_update += 1

        # ✅ Step 3: Suppress noisy unmatched detections before making new tracks
        for i, det in enumerate(detections):
            if i not in assigned_dets:
                suppress = False
                for trk in self.tracks.values():
                    if trk.time_since_update == 0:  # only active ones
                        dist = np.linalg.norm(np.array(trk.get_prediction()) - np.array(det))
                        if dist < self.match_threshold / 2:
                            suppress = True
                            break
                if not suppress:
                    self._start_new_track(det)

        # ✅ Step 4: Remove dead tracks
        self.tracks = {
            tid: trk for tid, trk in self.tracks.items()
            if trk.time_since_update <= self.max_age
        }

        return self._get_confirmed_tracks()


    def _start_new_track(self, detection):
        self.tracks[self.next_id] = KalmanTrack(self.next_id, detection)
        self.next_id += 1

    def _get_confirmed_tracks(self):
        return {
            tid: {
                'position': trk.get_prediction(),
                'trail': list(trk.positions)
            }
            for tid, trk in self.tracks.items()
            if trk.confirmed
        }




colors = {0: (0,0,255),1:(0,255,0),2:(255,0,0),3:(255,255,0)}
consumer = KafkaConsumer(
    TOPIC_NAME,
    bootstrap_servers=[KAFKA_BROKER],
    value_deserializer=lambda x: json.loads(x.decode('utf-8')),
    auto_offset_reset='latest',
    enable_auto_commit=True
)


def plot_bev_opencv_old(tracks, frame_num):
    bev_img = _bev.copy()

    for tid, (x, y) in tracks.items():
        x, y = int(x), int(y)
        cv2.circle(bev_img, (x, y), 5, colors[tid % len(colors)], -1)
        cv2.putText(bev_img, f'ID:{tid}', (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    cv2.putText(bev_img, f'Frame: {frame_num}', (10, 390),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)

    cv2.imshow("BEV Real-time", bev_img)
    cv2.waitKey(1)


def smooth_trail(trail, alpha=0.7):
    if len(trail) < 3:
        return trail
    smoothed = [trail[0]]
    for pt in trail[1:]:
        last = smoothed[-1]
        new_x = alpha * pt[0] + (1 - alpha) * last[0]
        new_y = alpha * pt[1] + (1 - alpha) * last[1]
        smoothed.append((new_x, new_y))
    return smoothed


def plot_bev_opencv(tracks, frame_num):
    bev_img = _bev.copy()

    for tid, data in tracks.items():
        x, y = int(data['position'][0]), int(data['position'][1])
        trail = smooth_trail(data['trail'])

        cv2.circle(bev_img, (x, y), 5, colors[tid % len(colors)], -1)
        cv2.putText(bev_img, f'ID:{tid}', (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        for j in range(1, len(trail)):
            x1, y1 = int(trail[j - 1][0]), int(trail[j - 1][1])
            x2, y2 = int(trail[j][0]), int(trail[j][1])
            cv2.line(bev_img, (x1, y1), (x2, y2), colors[tid % len(colors)], 2)

    cv2.putText(bev_img, f'Frame: {frame_num}', (10, 390),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)

    cv2.imshow("BEV Real-time", bev_img)
    cv2.waitKey(1)





def extract_message(message):
    return json.loads(message.value['customMessage'][0])
def extract_centroids(message):
    return message['clusterMeta']['centroids']


tracker = KalmanTrackerManager()
try:
    for _message in consumer:
        message = extract_message(_message)
        centroids = extract_centroids(message)


    
        if centroids:
            detections = [(pt['x'], pt['y']) for pt in centroids]
            confirmed_tracks = tracker.update(detections)

            plot_bev_opencv(confirmed_tracks, message['frame_num'])
      





except KeyboardInterrupt:
    print("Stopped by user")
finally:
    cv2.destroyAllWindows()
    consumer.close()


