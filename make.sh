#!/bin/bash

function getbazel(){
	LINE=`readlink -f /home/nghia/Desktop/tensorflow/bazel-bin/`

	POS1="_bazel_$USER/"
	STR=${LINE##*$POS1}

	BAZEL=${STR:0:32}

	echo $BAZEL
}



BAZEL=`getbazel`


IINCLUDE=" -I/home/nghia/Downloads/DS-master/eigen-eigen-5a0156e40feb " 


LLIBPATH=" -L/home/nghia/Desktop/tensorflow/bazel-bin/tensorflow -L/home/nghia/Downloads/DS-master"

rm tracker -rf


function BOPENMPHOG(){
	LLIBS="-lopencv_core -lopencv_imgproc  -lopencv_highgui -lFeatureGetter -lboost_system -lglog -lopencv_imgcodecs -lopencv_videoio "
	g++ --std=c++14 -O3 -fopenmp -o tracker $IINCLUDE $LLIBPATH  src/Ctracker.cpp src/fhog.cpp  src/HungarianAlg.cpp   src/kcftracker.cpp src/readfile.cpp src/track.cpp $LLIBS
}

BOPENMPHOG




