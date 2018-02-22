#pragma once
#include <iostream>
#include <vector>
#include <deque>
#include <memory>
#include <array>
#define USE_OCV_KCF 1
#if USE_OCV_KCF
#include <opencv2/tracking.hpp>
#endif
#include "Detection.h"
#include <Eigen/Dense>
#include "defines.h"


class CTrack
{
public:
	CTrack(const Point_t& pt,
	    	const CRegion& region,
	    	size_t trackID,
	    	std::vector<FEATURE> feature,
		std::vector<pairRectId> _pre_rects
	    );

	track_t CalcDist(const Point_t& pt) const;
	track_t CalcDistJaccard(const cv::Rect& r) const;

	void Update(const Point_t& pt, const CRegion& region, FEATURE feature, pairRectId Rect_Id);
	void Update(const Point_t& pt, const CRegion& region, pairRectId Rect_Id);
	size_t m_trackID;
	size_t m_skippedFrames;
	int update_time;
	Point_t m_point;   
	CRegion m_region; 
	std::vector<pairRectId> pre_rects;
        std::vector<FEATURE> fts;

};

typedef std::vector<std::unique_ptr<CTrack>> tracks_t;
