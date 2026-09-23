# Menggunakan base image dari dusty-nv (NVIDIA Jetson Community)
# Image ini sudah memiliki L4T (Linux for Tegra), ROS Noetic, dan PyTorch untuk CUDA Jetson.
# Mengapa Noetic? Karena Noetic menggunakan Python 3 secara default, sehingga tidak ada konflik dengan MediaPipe/PyTorch.
FROM dustynv/ros:noetic-pytorch-l4t-r32.7.1

# Set working directory
WORKDIR /workspace

# Perbarui GPG key ROS yang sudah expired
RUN apt-get update || true && \
  apt-get install -y --no-install-recommends curl gnupg2 && \
  curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - && \
  rm -rf /var/lib/apt/lists/*

# Install dependensi sistem tambahan (untuk OpenCV/Camera/ROS)
RUN apt-get update && apt-get install -y \
  v4l-utils \
  libv4l-dev \
  alsa-utils \
  python3-pip \
  git \
  && rm -rf /var/lib/apt/lists/*

# Install MediaPipe (Versi 0.8.5 CUDA GPU untuk Jetson Nano dengan HDMI dummy plug)
# HDMI dummy plug memungkinkan EGL/GPU aktif tanpa monitor fisik.
# PENTING: Jangan install opencv via pip karena akan kompilasi berjam-jam. Gunakan --no-deps.
RUN apt-get update && apt-get install -y wget unzip libffi-dev && \
  wget https://github.com/PINTO0309/mediapipe-bin/releases/download/v0.8.5/v0.8.5.zip && \
  unzip v0.8.5.zip && \
  pip3 install --upgrade pip setuptools wheel && \
  pip3 install --no-cache-dir "cryptography<3.5" && \
  pip3 install --no-cache-dir absl-py "attrs>=19.1.0" numpy "protobuf<4" flask requests pyserial waitress pywebpush && \
  pip3 install --no-deps --no-cache-dir v0.8.5/numpy119x/py36/mediapipe-0.8.5_cuda102-cp36-cp36m-linux_aarch64.whl && \
  rm -rf v0.8.5.zip v0.8.5 && \
  rm -rf /var/lib/apt/lists/*

# Pre-download model TFLite Lite (Google's storage link sudah mati, pakai mirror)
# MediaPipe 0.8.x akan mencoba download model saat pertama kali dijalankan, tapi URL-nya 404.
# Kita sediakan file-nya di sini agar tidak perlu download saat runtime.
RUN mkdir -p /usr/local/lib/python3.6/dist-packages/mediapipe/modules/pose_landmark/ && \
  wget -q -O /usr/local/lib/python3.6/dist-packages/mediapipe/modules/pose_landmark/pose_landmark_lite.tflite \
  https://storage.googleapis.com/mediapipe-assets/pose_landmark_lite.tflite && \
  wget -q -O /usr/local/lib/python3.6/dist-packages/mediapipe/modules/pose_landmark/pose_landmark_full.tflite \
  https://storage.googleapis.com/mediapipe-assets/pose_landmark_full.tflite || true

# Buat Catkin Workspace
RUN mkdir -p /root/catkin_ws/src
WORKDIR /root/catkin_ws

# Kita tidak lagi menggunakan usb_cam maupun vision_opencv C++

# Copy ROS package yang sudah kita buat ke dalam workspace container
COPY ./ros_packages/fall_detection_ros ./src/fall_detection_ros

# Compile ROS workspace
RUN chmod +x /root/catkin_ws/src/fall_detection_ros/scripts/*.py && \
  /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make"

# Copy source code project asli (model CNN-LSTM, config, dll) ke dalam container
# CATATAN: Folder data/ dan results/ TIDAK di-copy karena tidak diperlukan untuk inference
RUN mkdir -p /workspace/FallDetectionCNN/models \
  && mkdir -p /workspace/FallDetectionCNN/results/figures
COPY ./src /workspace/FallDetectionCNN/src
COPY ./models /workspace/FallDetectionCNN/models
COPY ./img /workspace/FallDetectionCNN/img
COPY ./scripts /workspace/FallDetectionCNN/scripts
# The version-2 checkpoint embeds its scaler. The desktop `.task` asset may also
# be present in models/, while MediaPipe 0.8.x on Jetson uses its bundled Lite model.

# Setup environment variables
ENV ROS_MASTER_URI=http://localhost:11311
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Beri akses eksekusi pada script python ROS Node
RUN chmod +x /root/catkin_ws/src/fall_detection_ros/scripts/*.py

# Script entrypoint: start roscore di background, source ROS setup, lalu jalankan CMD
RUN echo '#!/bin/bash\n\
  set -e\n\
  source /opt/ros/noetic/setup.bash\n\
  source /root/catkin_ws/devel/setup.bash\n\
  # Start roscore di background\n\
  roscore &\n\
  # Tunggu roscore siap (max 10 detik)\n\
  for i in $(seq 1 10); do\n\
  if rostopic list > /dev/null 2>&1; then break; fi\n\
  echo "Menunggu roscore... ($i/10)"\n\
  sleep 1\n\
  done\n\
  exec "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

# Default command: jalankan launch file
# src_path diset ke absolute path di dalam container
CMD ["roslaunch", "fall_detection_ros", "fall_detection.launch", "src_path:=/workspace/FallDetectionCNN"]
