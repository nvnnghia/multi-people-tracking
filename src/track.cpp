#include "track.h"
#define USE_OCV_KCF 1
///
CTrack::CTrack(
        const Point_t& pt,
        const CRegion& region,
        size_t trackID,
	std::vector<FEATURE> feature,
	std::vector<pairRectId> _pre_rects
        )
    :
	m_trackID(trackID),
	m_skippedFrames(0),
	update_time(1),
	m_point(pt),
	m_region(region)
{
	fts= feature;
	pre_rects = _pre_rects;
}

/// \brief CalcDist
track_t CTrack::CalcDist(const Point_t& pt) const
{
    Point_t diff = m_point - pt;
    return sqrtf(diff.x * diff.x + diff.y * diff.y);
}


track_t CTrack::CalcDistJaccard(const cv::Rect& r) const
{
    cv::Rect rr(m_region);

    track_t intArea = (r & rr).area();
    track_t unionArea = r.area() + rr.area() - intArea;

    return 1 - intArea / unionArea;
}

void CTrack::Update(
        const Point_t& pt,
        const CRegion& region,
	FEATURE feature,
	pairRectId Rect_Id
        )
{
	m_point= pt;
	m_region= region;
	fts.push_back(feature);
	pre_rects.push_back(Rect_Id);
	if(fts.size()>20)
	{
		fts.erase(fts.begin());
	}
}
void CTrack::Update(
        const Point_t& pt,
        const CRegion& region,
	pairRectId Rect_Id
        )
{
	m_point= pt;
	m_region= region;
	pre_rects.push_back(Rect_Id);
}


