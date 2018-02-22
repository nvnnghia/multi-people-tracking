# Multi people tracking
Multi people tracking using deep features
# video demo: 

MOT16-03: https://www.youtube.com/watch?v=X5nddFEgeJ0

MOT16-01: https://www.youtube.com/watch?v=CKRdOhWDROU

# Build

1. Download project sources
2. Install CMake
3. Install OpenCV (https://github.com/opencv/opencv) and OpenCV contrib (https://github.com/opencv/opencv_contrib) repositories
4. Install tensorflow
5. Build getfeature library.
    go to project folder
    cd getFeature
    ./make.sh
6. build project
    ./make.sh
7. run

Note: you need to replace some file's path in the make.sh

# Usage
  ./tracker video-file-name detection-file-name
# Thirdparty libraries
cpp deep_sort: C++ implementation of Simple Online Realtime Tracking with a Deep Association Metric: https://github.com/oylz/DS

Deep Association Metric: https://github.com/nwojke/deep_sort

Hungarian algorithm + Kalman filter multitarget tracker implementation: https://github.com/Smorodov/Multitarget-tracker#demo-videos

OpenCV (and contrib): https://github.com/opencv/opencv and https://github.com/opencv/opencv_contrib

Tensorflow: https://github.com/tensorflow/tensorflow
