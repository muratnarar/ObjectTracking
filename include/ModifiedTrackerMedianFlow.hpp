//
//  ModifiedTrackerMedianFlow.hpp
//  ObjectTracking
//
//  Created by Nuri Murat ARAR on 25.10.17.
//

#ifndef ModifiedTrackerMedianFlow_hpp
#define ModifiedTrackerMedianFlow_hpp

#include <stdio.h>

//
//  ModifiedTrackerMedianFlow.cpp
//  ObjectTracking
//
//  Created by Nuri Murat ARAR on 25.10.17.
//

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
//#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include <algorithm>
#include <limits.h>

using namespace cv;
using namespace std;
    
#undef MEDIAN_FLOW_TRACKER_DEBUG_LOGS
#ifdef MEDIAN_FLOW_TRACKER_DEBUG_LOGS
#define dprintf(x) printf x
#else
#define dprintf(x) do{} while(false)
#endif
    
/*
 *  ModifiedTrackerMedianFlow
 */
/*
 * TODO:
 * add "non-detected" answer in algo --> test it with 2 rects --> frame-by-frame debug in TLD --> test it!!
 * take all parameters out
 *              asessment framework
 *
 *
 * FIXME:
 * when patch is cut from image to compute NCC, there can be problem with size
 * optimize (allocation<-->reallocation)
 */

struct TrackerParams{
    int pointsInGrid;
    Size winSize;
    int maxLevel;
    TermCriteria termCriteria;
    Size winSizeNCC;
    int maxMedianLengthOfDisplacementDifference;
};


class ModifiedTrackerMedianFlow{
    
public:

    
    //Constructor
    ModifiedTrackerMedianFlow(){
        _params.pointsInGrid=10;
        _params.winSize = Size(3,3);
        _params.maxLevel = 5;
        _params.termCriteria = TermCriteria(TermCriteria::COUNT|TermCriteria::EPS,20,0.3);
        _params.winSizeNCC = Size(30,30);
        _params.maxMedianLengthOfDisplacementDifference = 10;
    }
    
    ModifiedTrackerMedianFlow(TrackerParams params){
        _params = params;
        
    }
    
    // Initialize tracker
    void init(const Rect2d &currentROI, Mat currentImage);
    
    // Update position based on the new frame
    bool update(Mat newImage, Rect2d & newROI);
    
    void setImage(Mat image){
        _image = image.clone(); //TODO: check whether assignment works!
    }
    void setROI(Rect2d roi){
        _roi = roi; //TODO: check whether assignment works!
    }
    void setTrackerParams(TrackerParams params){
        _params = params;
    }
    
    Mat getImage(){
        return _image;
    }
    Rect2d getROI(){
        return _roi;
    }
    TrackerParams getParams(){
        return _params;
    }
    
private:
    
    Mat _image;
    Rect2d _roi;
    TrackerParams _params;
    
    bool medianFlowImpl(Mat oldImage,Mat newImage,Rect2d& oldBox);
    Rect2d vote(const std::vector<Point2f>& oldPoints,const std::vector<Point2f>& newPoints,const Rect2d& oldRect,Point2f& mD);
    void check_FB(const std::vector<Mat>& oldImagePyr,const std::vector<Mat>& newImagePyr,
                  const std::vector<Point2f>& oldPoints,const std::vector<Point2f>& newPoints,std::vector<bool>& status);
    void check_NCC(const Mat& oldImage,const Mat& newImage,
                   const std::vector<Point2f>& oldPoints,const std::vector<Point2f>& newPoints,std::vector<bool>& status);
    //float dist(Point2f p1,Point2f p2);
    //std::string type2str(int type);
    //TrackerMedianFlow::Params params;
 
};


template<typename T>
T getMedian( const std::vector<T>& values );

template<typename T>
T getMedianAndDoPartition( std::vector<T>& values );


#endif /* ModifiedTrackerMedianFlow_hpp */
