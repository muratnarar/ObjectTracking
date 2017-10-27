//
//  ModifiedTrackerMedianFlow.cpp
//  ObjectTracking
//
//  Created by Nuri Murat ARAR on 25.10.17.
//

#include "ModifiedTrackerMedianFlow.hpp"


Mat getPatch(Mat image, Size patch_size, Point2f patch_center)
{
    Mat patch;
    Point2i roi_strat_corner(cvRound(patch_center.x - patch_size.width / 2.),
                             cvRound(patch_center.y - patch_size.height / 2.));
    
    Rect2i patch_rect(roi_strat_corner, patch_size);
    
    if(patch_rect == (patch_rect & Rect2i(0, 0, image.cols, image.rows)))
    {
        patch = image(patch_rect);
    }
    else
    {
        getRectSubPix(image, patch_size,
                      Point2f((float)(patch_rect.x + patch_size.width  / 2.),
                              (float)(patch_rect.y + patch_size.height / 2.)), patch);
    }
    
    return patch;
}

template<typename T>
size_t filterPointsInVectors(std::vector<T>& status, std::vector<Point2f>& vec1, std::vector<Point2f>& vec2, T goodValue)
{
    CV_DbgAssert(status.size() == vec1.size() && status.size() == vec2.size());
    
    size_t first_bad_idx = 0;
    while(first_bad_idx < status.size())
    {
        if(status[first_bad_idx] != goodValue)
            break;
        first_bad_idx++;
    }
    
    if (first_bad_idx >= status.size())
        return first_bad_idx;
    
    for(size_t i = first_bad_idx + 1; i < status.size(); i++)
    {
        if (status[i] != goodValue)
            continue;
        
        status[first_bad_idx] = goodValue;
        vec1[first_bad_idx] = vec1[i];
        vec2[first_bad_idx] = vec2[i];
        first_bad_idx++;
    }
    vec1.erase(vec1.begin() + first_bad_idx, vec1.end());
    vec2.erase(vec2.begin() + first_bad_idx, vec2.end());
    status.erase(status.begin() + first_bad_idx, status.end());
    
    return first_bad_idx;
}

void ModifiedTrackerMedianFlow::init(const Rect2d &currentROI, Mat currentImage){
    setImage(currentImage);
    setROI(currentROI);
}

bool ModifiedTrackerMedianFlow::update(Mat newImage, Rect2d& newROI){
    Mat oldImage = getImage();
    Rect2d oldROI = getROI(); //TODO: check whether its possible to convert Rect to Rect2d
    if(!medianFlowImpl(oldImage,newImage,oldROI))
        return false;
    
    newROI = oldROI; //TODO: check whether its possible to convert Rect to Rect2d
    setImage(newImage);
    setROI(newROI);
    return true;
}

bool ModifiedTrackerMedianFlow::medianFlowImpl(Mat oldImage,Mat newImage,Rect2d& oldBox){
    std::vector<Point2f> pointsToTrackOld,pointsToTrackNew;
    
    Mat oldImage_gray,newImage_gray;
    if (oldImage.channels() != 1)
        cvtColor( oldImage, oldImage_gray, COLOR_BGR2GRAY );
    else
        oldImage.copyTo(oldImage_gray);
    
    if (newImage.channels() != 1)
        cvtColor( newImage, newImage_gray, COLOR_BGR2GRAY );
    else
        newImage.copyTo(newImage_gray);
    
    TrackerParams params = getParams();
    //"open ended" grid
    for(int i=0;i<params.pointsInGrid;i++){
        for(int j=0;j<params.pointsInGrid;j++){
            pointsToTrackOld.push_back(Point2f((float)(oldBox.x+((1.0*oldBox.width)/params.pointsInGrid)*j+.5*oldBox.width/params.pointsInGrid),
                                               (float)(oldBox.y+((1.0*oldBox.height)/params.pointsInGrid)*i+.5*oldBox.height/params.pointsInGrid)));
        }
    }
    
    std::vector<uchar> status(pointsToTrackOld.size());
    std::vector<float> errors(pointsToTrackOld.size());
    
    std::vector<Mat> oldImagePyr;
    buildOpticalFlowPyramid(oldImage_gray, oldImagePyr, params.winSize, params.maxLevel, false);
    
    std::vector<Mat> newImagePyr;
    buildOpticalFlowPyramid(newImage_gray, newImagePyr, params.winSize, params.maxLevel, false);
    
    calcOpticalFlowPyrLK(oldImagePyr,newImagePyr,pointsToTrackOld,pointsToTrackNew,status,errors,
                         params.winSize, params.maxLevel, params.termCriteria, 0);
    
    CV_Assert(pointsToTrackNew.size() == pointsToTrackOld.size());
    CV_Assert(status.size() == pointsToTrackOld.size());
    dprintf(("\t%d after LK forward\n",(int)pointsToTrackOld.size()));
    
    size_t num_good_points_after_optical_flow = filterPointsInVectors(status, pointsToTrackOld, pointsToTrackNew, (uchar)1);
    
    dprintf(("\t num_good_points_after_optical_flow = %d\n",num_good_points_after_optical_flow));
    
    if (num_good_points_after_optical_flow == 0) {
        return false;
    }
    
    CV_Assert(pointsToTrackOld.size() == num_good_points_after_optical_flow);
    CV_Assert(pointsToTrackNew.size() == num_good_points_after_optical_flow);
    
    dprintf(("\t%d after LK forward after removing points with bad status\n",(int)pointsToTrackOld.size()));
    
    std::vector<bool> filter_status(pointsToTrackOld.size(), true);
    check_FB(oldImagePyr, newImagePyr, pointsToTrackOld, pointsToTrackNew, filter_status);
    check_NCC(oldImage_gray, newImage_gray, pointsToTrackOld, pointsToTrackNew, filter_status);
    
    // filter
    size_t num_good_points_after_filtering = filterPointsInVectors(filter_status, pointsToTrackOld, pointsToTrackNew, true);
    
    dprintf(("\t num_good_points_after_filtering = %d\n",num_good_points_after_filtering));
    
    if(num_good_points_after_filtering == 0){
        return false;
    }
    
    CV_Assert(pointsToTrackOld.size() == num_good_points_after_filtering);
    CV_Assert(pointsToTrackNew.size() == num_good_points_after_filtering);
    
    dprintf(("\t%d after LK backward\n",(int)pointsToTrackOld.size()));
    
    std::vector<Point2f> di(pointsToTrackOld.size());
    for(size_t i=0; i<pointsToTrackOld.size(); i++){
        di[i] = pointsToTrackNew[i]-pointsToTrackOld[i];
    }
    
    Point2f mDisplacement;
    oldBox=vote(pointsToTrackOld,pointsToTrackNew,oldBox,mDisplacement);
    
    std::vector<float> displacements;
    for(size_t i=0;i<di.size();i++){
        di[i]-=mDisplacement;
        displacements.push_back((float)sqrt(di[i].ddot(di[i])));
    }
    float median_displacements = getMedianAndDoPartition(displacements);
    dprintf(("\tmedian of length of difference of displacements = %f\n", median_displacements));
    if(median_displacements > params.maxMedianLengthOfDisplacementDifference){
        dprintf(("\tmedian flow tracker returns false due to big median length of difference between displacements\n"));
        return false;
    }
    
    return true;
}

Rect2d ModifiedTrackerMedianFlow::vote(const std::vector<Point2f>& oldPoints,const std::vector<Point2f>& newPoints,const Rect2d& oldRect,Point2f& mD){
    Rect2d newRect;
    Point2d newCenter(oldRect.x+oldRect.width/2.0,oldRect.y+oldRect.height/2.0);
    const size_t n=oldPoints.size();
    
    if (n==1) {
        newRect.x=oldRect.x;
        newRect.y=oldRect.y+newPoints[0].y-oldPoints[0].y;
        newRect.width=oldRect.width;
        newRect.height=oldRect.height;
        mD.x = newPoints[0].x;
        mD.y = newPoints[0].y-oldPoints[0].y;
        return newRect;
    }
    
    float xshift=0,yshift=0;
    std::vector<float> buf_for_location(n, 0.);
    for(size_t i=0;i<n;i++){  buf_for_location[i]=newPoints[i].y-oldPoints[i].y;  }
    yshift=getMedianAndDoPartition(buf_for_location);
    newCenter.y+=yshift;
    mD=Point2f((float)xshift,(float)yshift);
    
    dprintf(("xshift, yshift, scale = %f %f %f\n",xshift,yshift));
    newRect.x=newCenter.x-oldRect.width/2.0;
    newRect.y=newCenter.y-oldRect.height/2.0;
    newRect.width=oldRect.width;
    newRect.height=oldRect.height;
    dprintf(("rect old [%f %f %f %f]\n",oldRect.x,oldRect.y,oldRect.width,oldRect.height));
    dprintf(("rect [%f %f %f %f]\n",newRect.x,newRect.y,newRect.width,newRect.height));
    
    return newRect;
}

/*
Rect2d ModifiedTrackerMedianFlow::voteOrig(const std::vector<Point2f>& oldPoints,const std::vector<Point2f>& newPoints,const Rect2d& oldRect,Point2f& mD){
    Rect2d newRect;
    Point2d newCenter(oldRect.x+oldRect.width/2.0,oldRect.y+oldRect.height/2.0);
    const size_t n=oldPoints.size();
    
    if (n==1) {
        newRect.x=oldRect.x+newPoints[0].x-oldPoints[0].x;
        newRect.y=oldRect.y+newPoints[0].y-oldPoints[0].y;
        newRect.width=oldRect.width;
        newRect.height=oldRect.height;
        mD.x = newPoints[0].x-oldPoints[0].x;
        mD.y = newPoints[0].y-oldPoints[0].y;
        return newRect;
    }
    
    float xshift=0,yshift=0;
    std::vector<float> buf_for_location(n, 0.);
    for(size_t i=0;i<n;i++){  buf_for_location[i]=newPoints[i].x-oldPoints[i].x;  }
    xshift=getMedianAndDoPartition(buf_for_location);
    newCenter.x+=xshift;
    for(size_t i=0;i<n;i++){  buf_for_location[i]=newPoints[i].y-oldPoints[i].y;  }
    yshift=getMedianAndDoPartition(buf_for_location);
    newCenter.y+=yshift;
    mD=Point2f((float)xshift,(float)yshift);
    
    std::vector<double> buf_for_scale(n*(n-1)/2, 0.0);
    for(size_t i=0,ctr=0;i<n;i++){
        for(size_t j=0;j<i;j++){
            double nd=norm(newPoints[i] - newPoints[j]);
            double od=norm(oldPoints[i] - oldPoints[j]);
            buf_for_scale[ctr]=(od==0.0)?0.0:(nd/od);
            ctr++;
        }
    }
    
    double scale=getMedianAndDoPartition(buf_for_scale);
    dprintf(("xshift, yshift, scale = %f %f %f\n",xshift,yshift,scale));
    newRect.x=newCenter.x-scale*oldRect.width/2.0;
    newRect.y=newCenter.y-scale*oldRect.height/2.0;
    newRect.width=scale*oldRect.width;
    newRect.height=scale*oldRect.height;
    dprintf(("rect old [%f %f %f %f]\n",oldRect.x,oldRect.y,oldRect.width,oldRect.height));
    dprintf(("rect [%f %f %f %f]\n",newRect.x,newRect.y,newRect.width,newRect.height));
    
    return newRect;
}
 */

void ModifiedTrackerMedianFlow::check_FB(const std::vector<Mat>& oldImagePyr, const std::vector<Mat>& newImagePyr,
                                         const std::vector<Point2f>& oldPoints, const std::vector<Point2f>& newPoints, std::vector<bool>& status){
    
    if(status.empty()) {
        status=std::vector<bool>(oldPoints.size(),true);
    }
    
    std::vector<uchar> LKstatus(oldPoints.size());
    std::vector<float> errors(oldPoints.size());
    std::vector<float> FBerror(oldPoints.size());
    std::vector<Point2f> pointsToTrackReprojection;
    TrackerParams params = getParams();
    calcOpticalFlowPyrLK(newImagePyr, oldImagePyr,newPoints,pointsToTrackReprojection,LKstatus,errors,
                         params.winSize, params.maxLevel, params.termCriteria, 0);
    
    for(size_t i=0;i<oldPoints.size();i++){
        FBerror[i]=(float)norm(oldPoints[i]-pointsToTrackReprojection[i]);
    }
    float FBerrorMedian=getMedian(FBerror);
    dprintf(("point median=%f\n",FBerrorMedian));
    dprintf(("FBerrorMedian=%f\n",FBerrorMedian));
    for(size_t i=0;i<oldPoints.size();i++){
        status[i]=status[i] && (FBerror[i] <= FBerrorMedian);
    }
}

void ModifiedTrackerMedianFlow::check_NCC(const Mat& oldImage,const Mat& newImage,
                                          const std::vector<Point2f>& oldPoints,const std::vector<Point2f>& newPoints,std::vector<bool>& status){
    
    std::vector<float> NCC(oldPoints.size(),0.0);
    Mat p1,p2;
    TrackerParams params = getParams();
    
    for (size_t i = 0; i < oldPoints.size(); i++) {
        p1 = getPatch(oldImage, params.winSizeNCC, oldPoints[i]);
        p2 = getPatch(newImage, params.winSizeNCC, newPoints[i]);
        
        const int patch_area=params.winSizeNCC.area();
        double s1=sum(p1)(0),s2=sum(p2)(0);
        double n1=norm(p1),n2=norm(p2);
        double prod=p1.dot(p2);
        double sq1=sqrt(n1*n1-s1*s1/patch_area),sq2=sqrt(n2*n2-s2*s2/patch_area);
        double ares=(sq2==0)?sq1/abs(sq1):(prod-s1*s2/patch_area)/sq1/sq2;
        
        NCC[i] = (float)ares;
    }
    float median = getMedian(NCC);
    for(size_t i = 0; i < oldPoints.size(); i++) {
        status[i] = status[i] && (NCC[i] >= median);
    }
}

template<typename T>
T getMedian(const std::vector<T>& values)
{
    std::vector<T> copy(values);
    return getMedianAndDoPartition(copy);
}

template<typename T>
T getMedianAndDoPartition(std::vector<T>& values)
{
    size_t size = values.size();
    if(size%2==0)
    {
        std::nth_element(values.begin(), values.begin() + size/2-1, values.end());
        T firstMedian = values[size/2-1];
        
        std::nth_element(values.begin(), values.begin() + size/2, values.end());
        T secondMedian = values[size/2];
        
        return (firstMedian + secondMedian) / (T)2;
    }
    else
    {
        size_t medianIndex = (size - 1) / 2;
        std::nth_element(values.begin(), values.begin() + medianIndex, values.end());
        
        return values[medianIndex];
    }
}
