#include "Ctracker.h"
#include "HungarianAlg.h"
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "kcftracker.hpp"
#include "nn_matching.h"
#include "FeatureGetter.h"
#include <dirent.h>
// ---------------------------------------------------------------------------
using namespace cv;
using namespace Eigen;
using namespace std;
#define UDL
boost::shared_ptr<NearestNeighborDistanceMetric> NearestNeighborDistanceMetric::self_;
#ifdef UDL
void ExtractFeature(const cv::Mat &in, 
	const std::vector<cv::Rect> &rcsin,
	std::vector<FEATURE> &fts) {
	int maxw = 0;
	int maxh = 0;
	int count = rcsin.size(); 
	std::vector<cv::Mat> faces;
	cv::Rect lr;
	for (int i = 0; i < count; i++) {
		cv::Rect rc;
		if(i < rcsin.size()){
			rc = rcsin[i];
			lr = rc;
		}
		else{
			rc = lr;
		}
		faces.push_back(in(rc).clone());
		int w = rc.width;
		int h = rc.height;
		if (w > maxw) {
			maxw = w;
		}
		if (h > maxh) {
			maxh = h;
		}
	}
	maxw += 10;
	maxh += 10;

	cv::Mat frame(maxh, maxw*count, CV_8UC3);
	std::vector<cv::Rect> rcs;
	for (int i = 0; i < count; i++) {
		cv::Mat &face = faces[i];
		cv::Rect rc = cv::Rect(i*maxw + 5, 5, face.cols, face.rows);
		rcs.push_back(rc);
		cv::Mat tmp = frame(rc);
		face.copyTo(tmp);
	}
	std::vector<FEATURE> newfts;
	FeatureGetter::Instance()->Get(frame, rcs, newfts);
	for(int i = 0; i < rcsin.size(); i++){
		fts.push_back(newfts[i]);
	}
}
#endif
struct DDS{
    public:
	void Push(int pos, const Eigen::VectorXf &dd){
		boost::mutex::scoped_lock lock(mutex_);
		dds_.push_back(
			std::make_pair(pos, dd)
		);
	}
	void Get(std::vector<std::pair<int, Eigen::VectorXf> > &dds){
		dds = dds_;
	}
    private:
	std::vector<std::pair<int, Eigen::VectorXf> > dds_;
	boost::mutex mutex_;
    };
CTracker::CTracker(
        )
    :
      NextTrackID(0)
{
}

// ---------------------------------------------------------------------------
CTracker::~CTracker(void)
{
}

// ---------------------------------------------------------------------------
void CTracker::Update(
        const std::vector<Point_t>& detections,
        const regions_t& regions,
        cv::Mat grayFrame,
	cv::Mat frame,
	int frame_count
        )
{
	assert(detections.size() == regions.size());
	#ifdef UDL
	if(!FeatureGetter::Instance()->Init()){
		std::cout<<"=======================error init Feature getter"<<std::endl<<std::endl;
		return ;
	}
	#endif
	NearestNeighborDistanceMetric::Instance()->Init(0.2, 100);
	std::vector<FEATURE> fts;
	if(!detections.empty())
		ExtractFeature(frame, regions, fts);
	FEATURESS ftss(fts.size(), 128);
	for (int k = 0; k < fts.size(); k++) {
				ftss.row(k) = fts[k];
			}
	//calculate track feature distance
	
	size_t N = tracks.size();		
	size_t M = detections.size();
	static DYNAMICM cost_matrix;
    	if(!tracks.empty())
	{
		//static DYNAMICM cost_matrix;
		cost_matrix = DYNAMICM(tracks.size(), detections.size());
		DDS dds;
		for(int i =0; i<tracks.size();i++)
		{
			FEATURESS ftstrack(tracks[i]->fts.size(), 128);
			for (int k = 0; k < tracks[i]->fts.size(); k++) {
				ftstrack.row(k) = tracks[i]->fts[k];
			}
			Eigen::VectorXf dd = _nn_cosine_distance(ftstrack, ftss);
			dds.Push(i, dd);
			//std::cout<<" ===== ddd = "<<dd<<std::endl;
			
		}
		std::vector<std::pair<int, Eigen::VectorXf>> vec;
		dds.Get(vec);
		for(int i = 0; i < vec.size(); i++){
			std::pair<int, Eigen::VectorXf> pa = vec[i];
			cost_matrix.row(pa.first) =  pa.second;
		} 
		//std::cout << "\nb-haha\n" << cost_matrix << "\ne-haha\n";
	}
	//~calculate track feature distance
	regions_t new_regions;
        std::vector<Point_t> new_centers;
	std::vector<FEATURE> new_fts;

	regions_t remain_regions;
        std::vector<Point_t> remain_centers;
	std::vector<FEATURE> remain_fts;
    	assignments_t assignment(N, -1); 
	distMatrix_t Cost(N * M);
    	if (!tracks.empty())
    	{
		
		track_t maxCost = 0;
	    	for (size_t i = 0; i < tracks.size(); i++)
	    	{
			for (size_t j = 0; j < detections.size(); j++)
			{
			    	auto dist = tracks[i]->CalcDistJaccard(regions[j]);
				if ((dist >0.6f)||(cost_matrix(i,j)>0.2))
			    		Cost[i + j * N] = 2;
				else 
					Cost[i + j * N] = cost_matrix(i,j);
				
			    	if (Cost[i + j * N] > maxCost)
			    	{
					maxCost = Cost[i + j * N];
			    	}
			}
	    	}
		// Solving assignment problem 
		AssignmentProblemSolver APS;
		APS.Solve(Cost, N, M, assignment, AssignmentProblemSolver::many_forbidden_assignments);

		// clean assignment from pairs with large distance
		for (size_t i = 0; i < assignment.size(); i++)
		{
			if (assignment[i] != -1)
			{
				if (Cost[i + assignment[i] * N] > 0.99f)
				{
					assignment[i] = -1;
				}
			}
		}

    	}

// Update trackers state
	int trackup=0; int tracklostup=0; int tracknew=0;
    	for (size_t i = 0; i < assignment.size(); i++)
    	{
		if ((assignment[i] != -1)) // If we have assigned detect, then update using its coordinates,
		{
		    	tracks[i]->m_skippedFrames = 0;
			pairRectId sss(regions[assignment[i]],frame_count);
		    	tracks[i]->Update(detections[assignment[i]], regions[assignment[i]], fts[assignment[i]], sss);
			if(tracks[i]->pre_rects[0].frameId ==frame_count-19) tracks[i]->pre_rects.erase(tracks[i]->pre_rects.begin());
		} 
		else  // If we dont have assigned detect, then continue track using KCF or delete track if skipframe>10,
			
		if((tracks[i]->m_skippedFrames<0))
		{
			bool HOG = true;
			bool FIXEDWINDOW = true;
			bool MULTISCALE = false;
			bool LAB = true;
			KCFTracker track(HOG, FIXEDWINDOW, MULTISCALE, LAB);
			cv::Rect result;
			track.init( tracks[i]->m_region, m_prevFrame);
			result = track.update(frame);
			tracks[i]->m_skippedFrames+=2;
			assignment[i] = 1;
			Point_t centers((result.tl() + result.br()) / 2);
			CRegion region(result);

			/*vector<Rect> regionkcf;
			regionkcf.push_back(result);
			std::vector<FEATURE> ftskcf;
			ExtractFeature(frame, regionkcf, ftskcf);
			//std::cout<<"=======================cnn feature size "<<fts.size()<<fts[1].size()<<std::endl<<std::endl;
			FEATURESS ftsskcf(ftskcf.size(), 128);
			for (int k = 0; k < ftskcf.size(); k++) {
						ftsskcf.row(k) = ftskcf[k];
					}
			FEATURESS ftstrack(tracks[i]->fts.size(), 128);
			for (int k = 0; k < tracks[i]->fts.size(); k++) {
				ftstrack.row(k) = tracks[i]->fts[k];
			}
			Eigen::VectorXf dd = _nn_cosine_distance(ftstrack, ftsskcf);
			cout<<"result kcf feature = "<<dd<<" " <<dd[0]<<endl;*/
			if((tracks[i]->CalcDistJaccard(result)<0.3))
			{
				pairRectId sss(region,frame_count);
				tracks[i]->Update(centers, region, sss); 
				if(tracks[i]->pre_rects[0].frameId ==frame_count-19) tracks[i]->pre_rects.erase(tracks[i]->pre_rects.begin());
			}
			else 
			{
				tracks[i]->m_skippedFrames+=2;
			}
			//tracks[tracks.size()-1]->pre_bbox.push_back(region);
			
		} 
    	}
//~update tracker state
//Remove lost object from tracker. Put into pre tracks
    	for (size_t i = 0; i < (assignment.size()); i++)
    	{
		if (assignment[i] == -1) 
		{ 
		   	lost_tracks.push_back(std::make_unique<CTrack>(tracks[i]->m_point,
				                                      	tracks[i]->m_region,
				                                      	tracks[i]->m_trackID,
									tracks[i]->fts, tracks[i]->pre_rects ));

			lost_tracks[lost_tracks.size()-1]->m_skippedFrames=1;
		}
    	}
    	int ii = (assignment.size())-1;
    	for (int i = ii; i >=0; i--)
    	{
		if (assignment[i] == -1) 
		{ 
		    tracks.erase(tracks.begin() + i);
		}
    	}  
//~Remove lost object from tracker. Put into pre tracks
// -----------------------------------
//~Track current tracked objects
    // Search for unassigned detects to find the lost objects or consider to create a new track.
    // -----------------------------------
	
    	for (size_t i = 0; i < detections.size(); i++)
    	{
		if (find(assignment.begin(), assignment.end(), i) == assignment.end())
		{
			remain_regions.push_back(regions[i]);
			remain_centers.push_back(detections[i]);
			remain_fts.push_back(fts[i]);
		}
    	}
    //~ Search for unassigned detects to find the lost objects or consider to create a new track.
//Find lost objects
	FEATURESS ftss_lost(remain_fts.size(), 128);
	for (int k = 0; k < remain_fts.size(); k++) {
				ftss_lost.row(k) = remain_fts[k];
			}
	size_t N_lost = lost_tracks.size();		
	size_t M_lost = remain_regions.size();
	distMatrix_t Cost_lost(N_lost*M_lost);
	assignments_t assignment_lost(N_lost, -1);
	if(!lost_tracks.empty())
	{
		static DYNAMICM cost_matrix_lost;
		cost_matrix_lost = DYNAMICM(lost_tracks.size(), remain_regions.size());
		DDS dds_lost;
		for(int i =0; i<lost_tracks.size();i++)
		{
			FEATURESS ftstrack_lost(lost_tracks[i]->fts.size(), 128);
			for (int k = 0; k < lost_tracks[i]->fts.size(); k++) {
				ftstrack_lost.row(k) = lost_tracks[i]->fts[k];
			}
			Eigen::VectorXf dd = _nn_cosine_distance(ftstrack_lost, ftss_lost);
			dds_lost.Push(i, dd);
			
		}
		std::vector<std::pair<int, Eigen::VectorXf>> vec;
		dds_lost.Get(vec);
		for(int i = 0; i < vec.size(); i++){
			std::pair<int, Eigen::VectorXf> pa = vec[i];
			cost_matrix_lost.row(pa.first) =  pa.second;
		} 
		track_t maxCost = 0;
	    	for (size_t i = 0; i < lost_tracks.size(); i++)
	    	{
			int num_lost = frame_count - lost_tracks[i]->pre_rects[lost_tracks[i]->pre_rects.size()-1].frameId;
			for (size_t j = 0; j < remain_centers.size(); j++)
			{
			    	auto dist = lost_tracks[i]->CalcDist(remain_centers[j]);
				if ((dist >0.5*num_lost*lost_tracks[i]->m_region.width)||(cost_matrix_lost(i,j)>0.2)||(dist >2*lost_tracks[i]->m_region.width))
			    		Cost_lost[i + j * N_lost] = 2;
				else Cost_lost[i + j * N_lost] = cost_matrix_lost(i,j);
				
			    	if (cost_matrix_lost(i,j) > maxCost)
			    	{
					maxCost = cost_matrix_lost(i,j);
			    	}
			}
	    	}
		//~distance
		// Solving assignment problem 
		AssignmentProblemSolver APS;
		APS.Solve(Cost_lost, N_lost, M_lost, assignment_lost, AssignmentProblemSolver::many_forbidden_assignments);
		// clean assignment from pairs with large distance
		for (size_t i = 0; i < assignment_lost.size(); i++)
		{
			if (assignment_lost[i] != -1)
			{
				if (Cost_lost[i + assignment_lost[i] * N_lost] > 1)
				{
					assignment_lost[i] = -1;
				}
			}
		}
		//update to tracker
		for (size_t i = 0; i < assignment_lost.size(); i++)
	    	{
			if ((assignment_lost[i] != -1)) 
			{
				int numframe = frame_count - lost_tracks[i]->pre_rects[lost_tracks[i]->pre_rects.size()-1].frameId;
				for(int count = numframe-1; count>0;count--)
				{
					cv::Rect aaa;
					aaa.x = (lost_tracks[i]->m_region.x*count + remain_regions[assignment_lost[i]].x*(numframe-count))/numframe;
					aaa.y = (lost_tracks[i]->m_region.y*count + remain_regions[assignment_lost[i]].y*(numframe-count))/numframe;
					aaa.width = (lost_tracks[i]->m_region.width*count + remain_regions[assignment_lost[i]].width*(numframe-count))/numframe;
					aaa.height = (lost_tracks[i]->m_region.height*count + remain_regions[assignment_lost[i]].height*(numframe-count))/numframe;
					pairRectId sss(aaa,frame_count-count);
					lost_tracks[i]->pre_rects.push_back(sss);
					if(lost_tracks[i]->pre_rects[0].frameId==frame_count-19)
						lost_tracks[i]->pre_rects.erase(lost_tracks[i]->pre_rects.begin());
					
				}
				pairRectId ss(remain_regions[assignment_lost[i]],frame_count);
				lost_tracks[i]->pre_rects.push_back(ss);
				if(lost_tracks[i]->pre_rects[0].frameId==frame_count-19)
						lost_tracks[i]->pre_rects.erase(lost_tracks[i]->pre_rects.begin());
				
				tracks.push_back(std::make_unique<CTrack>(remain_centers[assignment_lost[i]],
						                              	remain_regions[assignment_lost[i]],
						                              	lost_tracks[i]->m_trackID,
										lost_tracks[i]->fts, lost_tracks[i]->pre_rects ));
				tracks[tracks.size()-1]->m_skippedFrames=1;
				tracks[tracks.size()-1]->update_time=lost_tracks[i]->update_time;
				cout<<" success = == cost= find lost sceess  "<<std::endl;
				assignment_lost[i] = -2;
			} 
			else  
			{
				assignment_lost[i] = 1;
				lost_tracks[i]->m_skippedFrames++;
				if(lost_tracks[i]->m_skippedFrames>19)
				{
					assignment_lost[i] = -1;
					//active_tracks.erase(std::remove(active_tracks.begin(), active_tracks.end(), lost_tracks[i]->m_trackID),active_tracks.end());
				}
				if(lost_tracks[i]->pre_rects[0].frameId==frame_count-19)
						lost_tracks[i]->pre_rects.erase(lost_tracks[i]->pre_rects.begin());
				/*if(lost_tracks[i]->pre_rects.size()==20)
				{
					pairRectId ss(cv::Rect(1, 1, 300, 300),frame_count);
					if(lost_tracks[i]->pre_rects.size()>19)
						lost_tracks[i]->pre_rects.erase(lost_tracks[i]->pre_rects.begin());
					lost_tracks[i]->pre_rects.push_back(ss);
				}*/
			}
	    	}
		//~update tracker state
	//Remove lost object if
	    	int ii = (assignment_lost.size())-1;
	    	for (int i = ii; i >=0; i--)
	    	{
			if (assignment_lost[i] == -1) 
			{ 
				cout<<" -1 remove lost onject out " <<endl;
			    	lost_tracks.erase(lost_tracks.begin() + i);
				
			}
			if(assignment_lost[i]==-2)
			{
				lost_tracks.erase(lost_tracks.begin() + i);
				cout<<"-2 remove lost onject " <<lost_tracks.size()<<" ASS= "<<assignment_lost[i]<<endl;
			}
	    	}  

		for (size_t i = 0; i < remain_centers.size(); i++)
	    	{
			for(int j=0;j<assignment_lost.size();j++)
				if ((assignment_lost[j] ==i)||(assignment_lost[j] ==-2))
					break;
				else if(j==assignment_lost.size()-1)
				{	
					new_regions.push_back(remain_regions[i]);
					new_centers.push_back(remain_centers[i]);
					new_fts.push_back(remain_fts[i]);
					cout<<" fail ================"<<i<<" "<<remain_regions[i].x<<endl;
				}
	    	}
		
	}else 
	{
		new_regions = remain_regions;
		new_centers = remain_centers;
		new_fts = remain_fts;
	}
	
//~Find lost objects
	
	
//add new track
	
	size_t N_new = new_tracks.size();		
	size_t M_new = new_centers.size();	
	assignments_t assignment_new(N_new, -1); 
	if (!new_tracks.empty())
	{
		distMatrix_t Cost_new(N_new * M_new);
		track_t maxCost = 0;
		    	for (size_t i = 0; i < new_tracks.size(); i++)
		    	{
				for (size_t j = 0; j < new_centers.size(); j++)
				{
				    	auto dist = new_tracks[i]->CalcDistJaccard(new_regions[j]);
				    	Cost_new[i + j * N_new] = dist;
				    	if (dist > maxCost)
				    	{
						maxCost = dist;
				    	}
				}
		    	}
		// Solving assignment problem 
		// -----------------------------------
			AssignmentProblemSolver APS;
			APS.Solve(Cost_new, N_new, M_new, assignment_new, AssignmentProblemSolver::many_forbidden_assignments);
			// clean assignment from pairs with large distance
			// -----------------------------------
			for (size_t i = 0; i < assignment_new.size(); i++)
			{
				if (assignment_new[i] != -1)
				{
					if (Cost_new[i + assignment_new[i] * N_new] > 0.99f)
					{
						assignment_new[i] = -1;
					} 
				}
			}
		for (size_t i = 0; i < assignment_new.size(); i++)
	    	{	 
			if (assignment_new[i] != -1) // If we have assigned detect, then update it to tracker, object found
			{   
				new_tracks[i]->update_time++;
			    	if (new_tracks[i]->update_time>3)  // a object is consider to be a new object if it appear in at least 3 frames
			    	{
				    	new_tracks[i]->m_skippedFrames = 0;
				   	if(new_tracks[i]->m_trackID==0) 
					{
						new_tracks[i]->m_trackID=++NextTrackID;
					} 
					//active_tracks.push_back(new_tracks[i]->m_trackID);
					pairRectId ss(new_regions[assignment_new[i]], frame_count);
					new_tracks[i]->pre_rects.push_back(ss);
					if(new_tracks[i]->pre_rects[0].frameId ==frame_count-19) new_tracks[i]->pre_rects.erase(new_tracks[i]->pre_rects.begin());
				    	tracks.push_back(std::make_unique<CTrack>(new_centers[assignment_new[i]],
						                              	new_regions[assignment_new[i]],
						                              	new_tracks[i]->m_trackID,
										new_tracks[i]->fts, new_tracks[i]->pre_rects ));
					//tracks[tracks.size()-1]->isNew = true;
					assignment_new[i] = -2;
			   	} 
				else {
					new_tracks[i]->m_skippedFrames+=1;
					if(new_tracks[i]->m_skippedFrames>7) assignment_new[i] = -1;
					//fill missing new tracks
					int numframe = frame_count - new_tracks[i]->pre_rects[new_tracks[i]->pre_rects.size()-1].frameId;
					for(int count = numframe-1; count>0;count--)
						{
							cv::Rect aaa;
							aaa.x = (new_tracks[i]->m_region.x*count + new_regions[assignment_new[i]].x*(numframe-count))/numframe;
							aaa.y = (new_tracks[i]->m_region.y*count + new_regions[assignment_new[i]].y*(numframe-count))/numframe;
							aaa.width = (new_tracks[i]->m_region.width*count + new_regions[assignment_new[i]].width*(numframe-count))/numframe;
							aaa.height = (new_tracks[i]->m_region.height*count + new_regions[assignment_new[i]].height*(numframe-count))/numframe;
							pairRectId sss(aaa,frame_count-count);
							new_tracks[i]->pre_rects.push_back(sss);
							if(new_tracks[i]->pre_rects[0].frameId==frame_count-19)
								new_tracks[i]->pre_rects.erase(new_tracks[i]->pre_rects.begin());
					
						}
					//~fill missing new tracks
					pairRectId sss(new_regions[assignment_new[i]],frame_count);
					new_tracks[i]->Update(new_centers[assignment_new[i]], new_regions[assignment_new[i]], new_fts[assignment_new[i]], sss);
					if(new_tracks[i]->pre_rects[0].frameId ==frame_count-19) new_tracks[i]->pre_rects.erase(new_tracks[i]->pre_rects.begin());
					}
	
			} 
			else //If dont have assigned detect, then remove the objects if skipframe>10
			{
				new_tracks[i]->m_skippedFrames+=1;
				if(new_tracks[i]->m_skippedFrames<10) assignment_new[i] = 1;
			}
	    	}
//~add new track
		int ii1 = (assignment_new.size())-1;
	    	for (int i = ii1; i >=0; i--)
	    	{
			if (assignment_new[i] == -1) 
			{
			    new_tracks.erase(new_tracks.begin() + i);
			}
			if (assignment_new[i] == -2) 
			{ 
			    	new_tracks.erase(new_tracks.begin() + i);
			}
	    	}  
		cout<<" detection size = " <<detections.size()<<" remain size = "<<remain_regions.size()<<" new size = "<<new_centers.size()<<" new track before size = "<<new_tracks.size()<<" track lost up = "<<tracklostup<<endl;
	
	    	// -----------------------------------
		// Search for unassigned detects and start to track them.
	    	for (size_t i = 0; i < new_centers.size(); i++)
	    	{

			for(int j=0;j<assignment_new.size();j++)
					if ((assignment_new[j] ==i)||(assignment_new[j] ==-2))
						break;
					else if(j==assignment_new.size()-1)
					{	
						std::vector<FEATURE> ab;
						ab.push_back(new_fts[i]);
						std::vector<pairRectId> _pre_rects;
						pairRectId sss(new_regions[i], frame_count);
						_pre_rects.push_back(sss);
					    	new_tracks.push_back(std::make_unique<CTrack>(new_centers[i],
								                      	new_regions[i],
								                      	0,
											ab, _pre_rects));
						cout<<"new==================="<<endl;
					}

	    	}
	}
	else 
	{
		for (size_t i = 0; i < new_centers.size(); i++)
	    	{
			std::vector<FEATURE> ab;
			ab.push_back(new_fts[i]);
			std::vector<pairRectId> _pre_rects;
			pairRectId sss(new_regions[i], frame_count);
			_pre_rects.push_back(sss);
		    	new_tracks.push_back(std::make_unique<CTrack>(new_centers[i],
					                      	new_regions[i],
					                      	0,
								ab, _pre_rects));

	    	}
	}
    	
		//~ Search for unassigned detects and start to track them.
	m_prevFrame = frame;
}






