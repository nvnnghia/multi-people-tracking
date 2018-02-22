#pragma once
#include <iostream>
#include <vector>
#include <memory>
#include <array>

#include "defines.h"
#include "track.h"

// ----------------------------------------------------------------------
class CTracker
{
public:
	CTracker(void);
	~CTracker(void);
	std::vector<pairRectId> out_tracks;
	tracks_t tracks;
	tracks_t lost_tracks;
	tracks_t new_tracks;
	void Update(const std::vector<Point_t>& detections, const regions_t& regions, cv::Mat grayFrame, cv::Mat frame, int fame_count);

private:
	size_t NextTrackID;
	cv::Mat m_prevFrame;
};
