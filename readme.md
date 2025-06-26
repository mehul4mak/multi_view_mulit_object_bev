# 📌 Multi-View Multi-Object Tracking with BEV Visualization using DeepStream

This project implements a **multi-camera, multi-object tracking (MCMT)** pipeline with a **Bird’s Eye View (BEV)** visualization overlay. Built using **DeepStream 7.1 in C++**, it transmits metadata over **Kafka**, which is consumed by a **Python-based BEV visualizer**.

Please note the following:
* Used need to arrange data for cam_139.mp4, cam_140.mp4, , cam_142.mp4, , cam_52.mp4. I will add them later via git lfs.
* BEV Viz.py have a scope of improvement and that may be commited in near future.
* FPS improvements can be done with custom CUDA kernal or offloading CPU intensive work (tracker).


---

## 🚀 Features

- 🔄 Multi-camera synchronization and fusion
- 🎯 Centralized multi-object tracking (MOT)
- 📡 Real-time metadata streaming over Kafka
- 🧠 DeepStream 7.1 C++ pipeline
- 📍 BEV (Bird’s Eye View) visualization in Python
- 🔌 Custom plugin support (`nvmsgconv` provided)
- 🎥 Uses object detection + ReID models

---

## 📁 Project Structure

    .
    ├── app# DeepStream C++ application
    ├── lib/ # Custom .so files (e.g., nvmsgconv)
    ├── models/ # Object detection and ReID models
    ├── bev_visualizer.py # Kafka-based BEV visualization
    ├── config files
    └── README.md


---

## ⚙️ Installation

### 1. Install required system packages

```bash
sudo apt-get update
sudo apt-get install -y \
  libgstreamer1.0-dev \
  libgstreamer-plugins-base1.0-dev \
  libopencv-dev

./user_additional_install.sh
```

### Running the Application
1. Start the DeepStream C++ App
```
./app    # Or use a provided run script
```

2. Run the BEV Visualizer (Python)
```
python3 bev_Viz_v2.py
```
Make sure your Kafka broker is running and accessible.

### Contact
Mehul Kumawat <br>
📧 mehulkumawat@icloud.com

