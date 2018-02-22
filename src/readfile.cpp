#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include "Ctracker.h"
#include <map>
FILE *fptr;
using namespace std;
using namespace cv;
void DrawTrack(cv::Mat frame,
                   int resizeCoeff,
                   const CTrack& track
                   );
void DrawData(cv::Mat frame,int m_fps);
void DrawData1(cv::Mat frame,int m_fps);
std::vector<cv::Scalar> m_colors;
std::unique_ptr<CTracker> m_tracker;
std::vector<pairRectId> real_rects;
  cv::VideoCapture cap1;
float ratio= 1;
int frame_count=1;
void DrawTrack1(int x, int i, const CTrack& track);
int main(int argc, char **argv){
 
  // Create a VideoCapture object and open the input file
  // If the input is the web camera, pass 0 instead of the video file name
  cv::VideoCapture cap; 
  cap.open((argv[1]));
  cap1.open((argv[1]));
  cv::VideoWriter writer;
  // Check if camera opened successfully
  if(!cap.isOpened()){
    cout << "Error opening video stream or file" << endl;
    return -1;
  }
//Read detection results from file
        ifstream in(argv[2]);
	string wt;
	
	getline(in, wt, ',');
	
        int m_fps =10;
	m_colors.push_back(cv::Scalar(255, 0, 0));
        m_colors.push_back(cv::Scalar(0, 255, 0));
        m_colors.push_back(cv::Scalar(0, 0, 255));
        m_colors.push_back(cv::Scalar(255, 255, 0));
        m_colors.push_back(cv::Scalar(0, 255, 255));
        m_colors.push_back(cv::Scalar(255, 0, 255));
        m_colors.push_back(cv::Scalar(255, 127, 255));
        m_colors.push_back(cv::Scalar(127, 0, 255));
        m_colors.push_back(cv::Scalar(127, 0, 127));
// Innit tracker
	m_tracker = std::make_unique<CTracker>();
//~init tracker
//Process frame
cv::Rect objectsave; 
fptr = fopen("out.txt", "w+");
  while(1)
	{
 	
    cv::Mat frame;
    cv::Mat grayFrame;
    // Capture frame-by-frame
    cap >> frame;
    
    if (frame.empty())
            {
		std::cout<<"======= finish tracking"<<std::endl;
		int x=1;
		for(int i =frame_count-18;i<frame_count;i++)
		{
			
			for (const auto& track : m_tracker->tracks)
			{
				DrawTrack1(x, i, *track);
			}
			for (const auto& track1 : m_tracker->lost_tracks)
			{
				DrawTrack1(x, i, *track1);
			}
			x++;
		}
                break;
            }
	std::cout<<"\n======= new frame =================="<<std::endl;
    	cv::resize(frame, frame, cv::Size(), ratio, ratio);
    	cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
    	std::vector<cv::Rect> foundRects;
	if (!writer.isOpened())
           {
              writer.open("video3d.mp4", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 5, frame.size(), true);}
    // If the frame is empty, break immediately
    	if (frame.empty())
      		break;
	int objNum=0; double total_area =0;
	std::vector<Point_t> centers;
        regions_t regions;
 	while(1)
		{ 
							//detected object 
			int a; 
			if(objNum==0){
			 a=frame_count;}
			else {
				getline(in, wt, ',');
				 a = atoi(wt.c_str());
			}
			if (a==frame_count)
			{	cv::Rect object;
				getline(in, wt, ',');
				getline(in, wt, ',');
				object.x = ratio*atoi(wt.c_str());
				getline(in, wt, ',');
				object.y = ratio*atoi(wt.c_str());
				getline(in, wt, ',');
				object.width=ratio*atoi(wt.c_str());
				getline(in, wt, ',');
				object.height= ratio*atoi(wt.c_str());
				if(object.x<0) object.x=0;
				if(object.y<0) object.y=0;
				if(object.x+object.width>frame.cols) object.width=frame.cols-object.x;
				if(object.y+object.height>frame.rows) object.height=frame.rows-object.y;
				foundRects.push_back(object);
				total_area += object.area();
				getline(in, wt); objNum++;
			} else break; 
		} 
	//filter using object size
		for(int i=0;i<foundRects.size();i++)
		{
			if((foundRects[i].area()<4*(total_area/objNum))&&(foundRects[i].area()>0.1*(total_area/objNum))&&(foundRects[i].width<foundRects[i].height)&&(foundRects[i].height<8*foundRects[i].width))
		    {
			//rectangle(frame, foundRects[i],cv::Scalar(0,255,0),3);
			regions.push_back(foundRects[i]);
		    }
		}
		vector<int> detId;
		int a = regions.size();
		for(int i=a-1;i>=0;i--)
		{
			for(int j=i-1;j>=0;j--)
			{
				if((regions[i]&regions[j]).area()>0.9*regions[i].area())
					regions.erase(regions.begin()+j);
					//detId.push_back(j);
				if((regions[i]&regions[j]).area()>0.9*regions[j].area())
					regions.erase(regions.begin()+i);
			}
		}
	//~filter using object size
	
	for (auto rect : regions)
        {
		//rectangle(frame, cvPoint(rect.x,rect.y),cvPoint(rect.x+rect.width, rect.y+rect.height),cv::Scalar(0,0,255),2);
		centers.push_back((rect.tl() + rect.br()) / 2);
        }
       m_tracker->Update(centers, regions, grayFrame, frame, frame_count);
       std::cout<<"dddddddddddd ======== tracker size = " <<m_tracker->tracks.size()<<std::endl;
    // Display the resulting frame
	if(frame_count>18)
	{
		cap1 >> frame;
		DrawData(frame, m_fps);
		cv::imshow("Video", frame);
		
		if (writer.isOpened())
        	    {
         	      writer << frame;
          	 }
	}frame_count++;
	//if(frame_count==1194) break;
    char c=(char)cv::waitKey(1);
    if(c==27)
      break;
  }
  cap.release();
  in.close();
  cv::destroyAllWindows();
  fclose(fptr);
  return 0;
}
void DrawData(cv::Mat frame, int m_fps)
    {
	/*for (const auto& track : m_tracker->tracks)
        {
                if(track->isNew)
		{
			fprintf(fptr,"%d, %d, %d, %d, %d, %d, -1, -1, -1, -1\n",(frame_count-1), track->m_trackID, track->pre_rect.x, track->pre_rect.y, track->pre_rect.width,track->pre_rect.height );
		}
		std::cout<<"new track " <<track->isNew<<std::endl;
        }*/
	/*for (int i=0;i< m_tracker->active_tracks.size();i++)
        {
                std::cout<<" "<<m_tracker->active_tracks[i];
        }*/
        for (const auto& track : m_tracker->tracks)
        {
                DrawTrack(frame, 1, *track);
        }
	for (const auto& track1 : m_tracker->lost_tracks)
        {
                DrawTrack(frame, 1, *track1);
        }
    }
void DrawTrack(cv::Mat frame,
                   int resizeCoeff,
                   const CTrack& track
                   )
    {
	if(track.pre_rects[0].frameId==(frame_count-18))
	{
		cv::Scalar cl = m_colors[track.m_trackID % m_colors.size()];
		cv::Rect out_rect;
		out_rect.x = track.pre_rects[0].rect.x;
		out_rect.y = track.pre_rects[0].rect.y;
		out_rect.width = track.pre_rects[0].rect.width;
		out_rect.height= track.pre_rects[0].rect.height;
		string s = to_string(track.m_trackID);
		cv::rectangle(frame, out_rect , cl, 2, CV_AA);
		cv::Point point; point.x = track.pre_rects[0].rect.x; point.y = track.pre_rects[0].rect.y;
		putText(frame, s, cvPoint(track.pre_rects[0].rect.x-5, track.pre_rects[0].rect.y-5), CV_FONT_NORMAL, 1, cl,1,1);
		fprintf(fptr,"%d, %d, %d, %d, %d, %d, -1, -1, -1, -1\n",(frame_count-18), track.m_trackID, track.pre_rects[0].rect.x, track.pre_rects[0].rect.y, track.pre_rects[0].rect.width,track.pre_rects[0].rect.height );
	}
	
    }

void DrawTrack1(int x, int i, const CTrack& track)
    {
	if(track.pre_rects.size()>x)
	if(track.pre_rects[x].frameId==i)
	{
		fprintf(fptr,"%d, %d, %d, %d, %d, %d, -1, -1, -1, -1\n",i, track.m_trackID, track.pre_rects[x].rect.x, track.pre_rects[x].rect.y, track.pre_rects[x].rect.width,track.pre_rects[x].rect.height );
	}
	
    }











