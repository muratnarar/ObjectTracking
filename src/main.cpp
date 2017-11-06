//
//  main.cpp
//  ObjectTracking
//
//  Created by Nuri Murat ARAR on 09.10.17.
//  Copyright © 2017 Nuri Murat ARAR. All rights reserved.
//

#include <iostream>
#include <numeric>
#include <stdio.h>

// file handling
#include <boost/filesystem.hpp>
using namespace boost::filesystem;

// xml parsing
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/foreach.hpp>
using boost::property_tree::ptree;

// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>

//dlib
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/dir_nav.h>
#include <dlib/opencv.h>

// Custom tracker
#include "ModifiedTrackerMedianFlow.hpp"

// KCF tracker [https://github.com/joaofaro/KCFcpp]
#include "kcftracker.hpp"

using namespace cv;
using namespace std;
//using namespace dlib;


struct ImageSize{
    int  width, height, depth;
};

struct Object{
    string name, pose;
    int truncated, difficult, id, autodetect;
    Rect2d bounding_box;
};

struct Annotations{
    string folder_name, file_name, path;
    //set<string> source;
    ImageSize image_size;
    //int segmented;
    vector<Object> objects;
    Object object;
    bool load(const string &filename, int objectID);
    void save(const string &filename);
};

// Loads annotations structure from the specified XML file
bool Annotations::load(const string &filename, int objectID){
    // Create an empty property tree object
    ptree pt;
    
    // Load the XML file into the property tree. If reading fails
    // (cannot open file, parse error), an exception is thrown.
    read_xml(filename, pt);
    
    try{
    
        folder_name = pt.get<string>("annotation.folder");
        file_name = pt.get<string>("annotation.filename");
        path = pt.get<string>("annotation.path");
        
        image_size.width = pt.get<int>("annotation.size.width");
        image_size.height = pt.get<int>("annotation.size.height");
        image_size.depth = pt.get<int>("annotation.size.depth");
        
        BOOST_FOREACH (boost::property_tree::ptree::value_type& v, pt.get_child("annotation"))
        {
            if( v.first == "object"){
                if(v.second.get<int>("id") == objectID){
                    //segmented = pt.get<int>("annotation.segmented");
                    object.name = v.second.get<string>("name");
                    object.pose = v.second.get<string>("pose");
                    object.truncated = v.second.get<int>("truncated");
                    object.difficult = v.second.get<int>("difficult");
                    object.id = v.second.get<int>("id");
                    object.autodetect = v.second.get<int>("autodetect");
                    Point2d p1(v.second.get<double>("bndbox.xmin"),v.second.get<double>("bndbox.ymin"));
                    Point2d p2(v.second.get<double>("bndbox.xmax"),v.second.get<double>("bndbox.ymax"));
                    object.bounding_box = Rect(p1, p2);
                }
            }
        }
    }
    catch(const boost::property_tree::ptree_error &e){
        cout<<"There is a problem with loading of the annotation for file: "<<filename<<endl;
        cout<<e.what()<<endl;
        return false;
    }
    
    return true;
}

// return the filenames of all files that have the specified extension
// in the specified directory and all subdirectories
void getAllPathsWithExtension(string dir_path, string ext, vector<string>& ret){
    
    path root(dir_path);
    if(!exists(root) || !is_directory(root)){
        cout<<"Directory: "<<root<<" does not exist!"<<endl;
        return;
    }
    
    recursive_directory_iterator it(root);
    recursive_directory_iterator endit;
    
    while(it != endit)
    {
        if(is_regular_file(*it) && it->path().extension() == ext) ret.push_back(it->path().filename().string());
        ++it;
        
    }
    
}

vector<string> scanAllFiles(string test_dir_path, string file_extension){
    vector<string> image_paths;
    getAllPathsWithExtension(test_dir_path, file_extension, image_paths);
    sort(image_paths.begin(), image_paths.end());
    
    //for(int i = 0; i < image_paths.size(); i++)
    //    cout<<image_paths[i]<<endl;
    
    return image_paths;
    
}

Rect2d dlibRectangleToOpenCV(dlib::rectangle r){
    return Rect(Point2d(r.left(), r.top()), Point2d(r.right(), r.bottom()));
}

dlib::rectangle openCVRectToDlib(Rect r){
    //return dlib::rectangle((long)r.tl().x, (long)r.tl().y, (long)r.br().x - 1, (long)r.br().y - 1); //check dlib and opencv rect have different boundaries (shift by 1)
    return dlib::rectangle((long)r.tl().x, (long)r.tl().y, (long)r.br().x, (long)r.br().y);

}

Ptr<cv::Tracker> createInitializeOpenCVTracker(Mat &im, string tracker_type, Rect boundingBox){
    Ptr<cv::Tracker> tracker;

    if (tracker_type == "BOOSTING")
        tracker = TrackerBoosting::create();
    if (tracker_type == "MIL")
        tracker = TrackerMIL::create();
    if (tracker_type == "KCF")
        tracker = TrackerKCF::create();
    if (tracker_type == "TLD")
        tracker = TrackerTLD::create();
    if (tracker_type == "MEDIANFLOW"){
        tracker = TrackerMedianFlow::create();

        //If we want to change the default parameters
        //TrackerMedianFlow::Params params = TrackerMedianFlow::Params();
        //params.pointsInGrid = 2;
        //params.winSizeNCC = Size(50,50);

        /*
        TrackerMedianFlow::Params::Params() {
            pointsInGrid=10;
            winSize = Size(3,3);
            maxLevel = 5;
            termCriteria = TermCriteria(TermCriteria::COUNT|TermCriteria::EPS,20,0.3);
            winSizeNCC = Size(30,30);
            maxMedianLengthOfDisplacementDifference = 10;
        }*/
        //tracker = TrackerMedianFlow::create(params);
    }
    if (tracker_type == "GOTURN")
        tracker = TrackerGOTURN::create();
    
    tracker->init(im, boundingBox);
    
    return tracker;
}


double computeNCC(const Mat& oldROI, const Mat& newROI){
    
    const int patch_area=oldROI.rows*oldROI.cols;
    double s1=sum(oldROI)(0),s2=sum(newROI)(0);
    double n1=norm(oldROI),n2=norm(newROI);
    double prod=oldROI.dot(newROI);
    double sq1=sqrt(n1*n1-s1*s1/patch_area),sq2=sqrt(n2*n2-s2*s2/patch_area);
    double ares=(sq2==0)?sq1/abs(sq1):(prod-s1*s2/patch_area)/sq1/sq2;
    return ares;
}

// Exhaustive search for the best fit
// Consider only movement over a single axis and direction (+y axis direction)
// Typical behavior of the NCC value: first increase, get its peak, and decrease...
// The peak should ideally give the displacement
int findDisplacementByNCC(Mat &old_image, Mat &new_image, Rect2d &roi){
    
    int max_search_shift = 20;
    double old_ncc = -2, tmp_ncc, max_ncc = -1;
    float final_shift = -1;
    int breaker = 0;
    for(int shift_y = 0; shift_y < max_search_shift; shift_y++){
        tmp_ncc = computeNCC(old_image(roi),new_image(Rect2d(roi.x,roi.y+shift_y,roi.width,roi.height)));
        if(tmp_ncc > max_ncc){
            max_ncc = tmp_ncc;
            final_shift = shift_y;
        }
        //cout<<"\t shift_y:"<<shift_y<<"\t tmp_ncc:"<<tmp_ncc<<endl;
        
        //Try to break the loop
        
        // Simple idea1: if contionous decrease for more than twice, break
        // Not ideal in some cases. Gives erronous breaks at the beginning of the loop.
        //if(tmp_ncc < old_ncc)
        //    breaker++;
        //if(breaker == 2)
        //    break;
        
        // Simple idea2: if the change wrt max_ncc grows bigger, break
        //change = max_ncc - tmp_ncc;
        //if(change > change_thr){
        //    break;
        //}
        
        //old_ncc = tmp_ncc;
    }
    //cout<<"Shift y: "<<final_shift<<" \t max_ncc:"<<max_ncc<<endl;
    
    return final_shift;
    
}

// Calculate the displacement by running the trackers
// Initialize the tracker on the first frame (old_image) on the given roi
// Track the roi on the second image (new_image)
// Compute and return the displacements (shift_x, shift_y, shift_total) between the rois
void findDisplacementByTrackers(Mat &old_image, Mat &new_image, string tracker_type, Rect2d &roi, float &shift_x, float &shift_y, float &shift_total){
    Ptr<cv::Tracker> cv_tracker;
    ModifiedTrackerMedianFlow mmf_tracker;
    dlib::correlation_tracker dlib_tracker;
    KCFTracker kcf_tracker(true,false,true,true);
    
    Point2f centerGT = (roi.br()+ roi.tl())*0.5;

    //Initialize the tracker with the old_im
    if(tracker_type == "MODIFIED_MEDIANFLOW")
        mmf_tracker.init(roi,old_image);
    else if(tracker_type == "GITHUB-KCF")
        kcf_tracker.init(roi,old_image);
    else if(tracker_type == "DLIB_CORRELATION"){
        dlib::array2d<unsigned char> dlibImageGray;
        dlib::assign_image(dlibImageGray, dlib::cv_image<unsigned char>(old_image));
        dlib::rectangle droi = openCVRectToDlib(roi);
        dlib_tracker.start_track(dlibImageGray, droi);
    }
    else
        cv_tracker = createInitializeOpenCVTracker(old_image,tracker_type,roi);
    
    Rect2d new_box;
    dlib::rectangle new_roi;
    
    if(tracker_type == "MODIFIED_MEDIANFLOW")
        mmf_tracker.update(new_image,new_box);
    else if(tracker_type == "GITHUB-KCF")
        new_box = kcf_tracker.update(new_image);
    else if(tracker_type == "DLIB_CORRELATION"){
        dlib::array2d<unsigned char> dlibImageGray;
        dlib::assign_image(dlibImageGray, dlib::cv_image<unsigned char>(new_image));
        dlib_tracker.update(dlibImageGray);
        if(dlib_tracker.get_position().is_empty() == false){
            new_roi  = dlib_tracker.get_position();
            new_box = dlibRectangleToOpenCV(new_roi);
        }
    }
    else
        cv_tracker->update(new_image,new_box);
    
    Point2f centerTrackingResult = (new_box.br()+ new_box.tl())*0.5;

    //calculate the shifts
    shift_x = abs(centerTrackingResult.x-centerGT.x);
    shift_y = abs(centerTrackingResult.y-centerGT.y);
    shift_total = norm(centerTrackingResult-centerGT);
    
}

/*
 Evaluation of a given method in terms of displacement over a fixed observation window

 Input:
 -test_dir_path: <string> path to the test folder containing images and annotations (.xml files)
 -file_extension: <string> extension of the images e.g., ".png", ".jpg" etc.
 -frame_begin: <int> number of the first image to be processed (-1 for no particular range of images)
 -frame_end: <int> number of the last image to be processed (-1 for no particular range of images)
 -tracker_type: <string> tracker type e.g., "NCC", "MEDIANFLOW", "BOOSTING", "KCF", etc.
 -objectID: <int> id of the object. is not required to compute the displacement, but for only to fetch the ground truth data
 -window <Rect2d> observation window on which the displacement is computed
 -report: <File*> pointer to the report file
 Output:
 -return whether the evaluation is successfully completed (true) or not (false)
 
 Comments: auto_detect flag is not considered since we assume that all annotations are visually verified
 */
bool evaluateDisplacement(string test_dir_path, string file_extension, int frame_begin, int frame_end, string tracker_type, int objectID, Rect2d window, FILE *report){
    
    if(objectID == -1){
        cout<<"Evaluation could not start! Please specify one of the objects..."<<endl;
        return false;
    }
    
    //Fetch the image paths
    vector<string> image_paths;
    
    image_paths = scanAllFiles(test_dir_path, file_extension);
    file_extension = ".xml";
    
    int no_of_images = image_paths.size();
    if(no_of_images <= 0){
        cout<<"Number of images is <= 0..."<<endl;
        return false;
    }
    else
        cout<<"Number of images in the test directory: "<<no_of_images<<endl;
    
    path path_annotation;
    string path_annot = "";
    
    Annotations gt, trackingResult;
    Point2d oldCenterGT, centerGT, centerTrackingResult;
    float fps;
    vector<float> fpss, shift_xs, shift_ys, shift_totals, dispGT_xs, dispGT_ys, dispGT_totals, disp_errors_x, disp_errors_y, disp_errors_total;
    float dispGT_x, dispGT_y, dispGT_total, disp_error_x, disp_error_y, disp_error_total;
    float shift_x = -1, shift_y = -1, shift_total = -1;
    Mat old_im,im, im_display, im_display2;
    int frame_no;
    Rect2d shifted_GT_roi, old_im_gt_roi;
    
    fprintf(report,"First Image Frame No,First Image Frame No,GT Disp(x),GT Disp(y),GT Disp(total),Disp(x),Disp(y),Disp(total),Error Disp(x),Error Disp(y),Error Disp(total),FPS,,GT ROI Rect (x),GT ROI Rect (y),GT ROI Rect (width),GT ROI Rect (height),Estimated ROI Rect (x),Estimated ROI Rect (y),Estimated ROI Rect (width),Estimated ROI Rect (height) \n");
    
    if(frame_begin == -1)
        frame_begin = 0;
    if(frame_end == -1)
        frame_end = no_of_images;
    
    int frame_counter;
    for(frame_counter = frame_begin; frame_counter < frame_end; frame_counter++){
        
        //read the image
        im = imread(test_dir_path+image_paths[frame_counter]);
        
        //get image frame no
        size_t found = image_paths[frame_counter].find_last_of("_");
        frame_no = stoi(image_paths[frame_counter].substr(found+1));
        
        //display purposes
        im_display = im.clone();
        // Observation window
        rectangle(im_display, window, Scalar(255,0,0), 2, 1 );
        
        //display the frame
        imshow(tracker_type, im_display);
        
        //read the ground truth
        path_annot = image_paths[frame_counter].substr(0, image_paths[frame_counter].size()-4);
        path_annot = test_dir_path + path_annot + ".xml";
        path path_annotation(path_annot);
        if(!exists(path_annotation)){
            cout<<"File: "<<path_annotation<<" does not exist!"<<endl;
            fprintf(report,"%d,%d,%d,%d,%d,%d,%d,%d, , \n",frame_no-1,frame_no,-1,-1,-1,-1,-1,-1);
            continue;
        }
        
        //if there is a problem with loading the annotation, skip the file
        if(!gt.load(path_annot, objectID))
            continue;
        
        centerGT = (gt.object.bounding_box.br()+ gt.object.bounding_box.tl())*0.5;
        // GT ROI
        rectangle(im_display, gt.object.bounding_box, Scalar(0,255,0), 2, 1 );
        
        // Start timer
        double timer = (double)getTickCount();
        
        if(frame_counter > 0){
            
            shift_x = 0;
            shift_y = 0;
            shift_total = 0;
            
            if(tracker_type == "NCC") // Exhaustive search for the best fit // Consider only movement over a single axis (y axis)
                shift_y = findDisplacementByNCC(old_im, im, window);
            else // Displacement estimation by tracking algorithms
                findDisplacementByTrackers(old_im, im, tracker_type, window, shift_x, shift_y, shift_total);

            shift_xs.push_back(shift_x);
            shift_totals.push_back(shift_total);
            shift_ys.push_back(shift_y);

            fps = getTickFrequency() / ((double)getTickCount() - timer);
            fpss.push_back(fps);
            
            //Calculate GT displacements
            dispGT_x = abs(centerGT.x-oldCenterGT.x);
            dispGT_y = abs(centerGT.y-oldCenterGT.y);
            dispGT_total = norm(centerGT-oldCenterGT);
            dispGT_ys.push_back(dispGT_y);
            dispGT_xs.push_back(dispGT_x);
            dispGT_totals.push_back(dispGT_total);
            
            //Calculate displacement errors
            disp_error_x = abs(shift_x-dispGT_x);
            disp_error_y = abs(shift_y-dispGT_y);
            disp_error_total = norm(Point2f(shift_x,shift_y)-Point2f(dispGT_x,dispGT_y));
            disp_errors_x.push_back(disp_error_x);
            disp_errors_y.push_back(disp_error_y);
            disp_errors_total.push_back(disp_error_total);
   
            shifted_GT_roi = Rect2d(old_im_gt_roi.x+shift_x, old_im_gt_roi.y+shift_y, old_im_gt_roi.width, old_im_gt_roi.height);

            fprintf(report,"%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,\n",frame_no-1,frame_no,dispGT_x,dispGT_y,dispGT_total,shift_x,shift_y,shift_total,disp_error_x,disp_error_y,disp_error_total,fps,gt.object.bounding_box.x,gt.object.bounding_box.y,gt.object.bounding_box.width,gt.object.bounding_box.height,shifted_GT_roi.x,shifted_GT_roi.y,shifted_GT_roi.width,shifted_GT_roi.height);

            cout<<frame_no-1<<"\t"<<frame_no<<"\t Estimated displacement by "<<tracker_type<<" -> (x):"<<shift_x<<", (y):"<<shift_y<<", (total):"<<shift_total<<"\t GT displacement -> (x):"<<dispGT_x<<", (y):"<<dispGT_y<<", (total):"<<dispGT_total<<endl;
            cout<<"\t\t Error "<<" -> (x):"<<disp_error_x<<", (y):"<<disp_error_y<<", (total):"<<disp_error_total<<endl<<endl<<endl;

            rectangle(im_display, shifted_GT_roi, Scalar(0,0,255), 2, 1 );

            //putText(im_display, "NCC output: " + to_string(ncc), Point(100,80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,255),2);
            putText(im_display, "FPS : " + to_string(int(fps)), Point(100,50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50), 2);
            
            //hconcat(oldROIPatch, newROIPatch, im_display2);
            //imshow("Images", im_display2);
        }
        
        // Display frame.
        imshow(tracker_type, im_display);
        old_im = im.clone();
        oldCenterGT = centerGT;
        old_im_gt_roi = gt.object.bounding_box;
        
        // Exit if ESC pressed.
        int k = waitKey(10);
        if(k == 27)
            break;
        
    }
    
    double sum_fps = accumulate(fpss.begin(), fpss.end(), 0.0);
    double mean_fps = sum_fps / fpss.size();
    
    cout<<endl<<"No of processed frames: "<<frame_counter<<endl;
    cout<<"Average speed (in fps): "<<mean_fps<<endl;
    
    double sum_shift_x = accumulate(shift_xs.begin(),shift_xs.end(), 0.0);
    double mean_shift_x = sum_shift_x/shift_xs.size();
    double sum_shift_total = accumulate(shift_totals.begin(),shift_totals.end(), 0.0);
    double mean_shift_total = sum_shift_total/shift_totals.size();
    double sum_shift_y = accumulate(shift_ys.begin(),shift_ys.end(), 0.0);
    double mean_shift_y = sum_shift_y/shift_ys.size();
    cout<<"Displacement estimated by "<<tracker_type<<" -> mean(x): "<<mean_shift_x<<", mean(y): "<<mean_shift_y<<", mean(total): "<<mean_shift_total<<endl;

    
    double sum_disp_GTx = accumulate(dispGT_xs.begin(), dispGT_xs.end(), 0.0);
    double mean_disp_GTx = sum_disp_GTx/dispGT_xs.size();
    double sum_disp_GT_total = accumulate(dispGT_totals.begin(), dispGT_totals.end(), 0.0);
    double mean_disp_GT_total = sum_disp_GT_total/dispGT_totals.size();
    double sum_disp_GTy = accumulate(dispGT_ys.begin(), dispGT_ys.end(), 0.0);
    double mean_disp_GTy = sum_disp_GTy/dispGT_ys.size();
    cout<<"Displacement on GT -> mean(x): "<<mean_disp_GTx<<", mean(y): "<<mean_disp_GTy<<", mean(total): "<<mean_disp_GT_total<<endl<<endl;

    
    double sum_disp_error_x = accumulate(disp_errors_x.begin(), disp_errors_x.end(), 0.0);
    double mean_disp_error_x = sum_disp_error_x/disp_errors_x.size();
    double sum_disp_error_total = accumulate(disp_errors_total.begin(), disp_errors_total.end(), 0.0);
    double mean_disp_error_total = sum_disp_error_total/disp_errors_total.size();
    double sum_disp_error_y = accumulate(disp_errors_y.begin(), disp_errors_y.end(), 0.0);
    double mean_disp_error_y = sum_disp_error_y/disp_errors_y.size();

    cout<<"Displacement error -> mean(x): "<<mean_disp_error_x<<", mean(y): "<<mean_disp_error_y<<", mean(total): "<<mean_disp_error_total<<endl;
    
    return true;
    
}


/*
 Evaluation of a given tracker for a given object
 Input:
 -test_dir_path: <string> path to the test folder containing images and annotations (.xml files)
 -file_extension: <string> extension of the images e.g., ".png", ".jpg" etc.
 -frame_begin: <int> number of the first image to be processed (-1 for no particular range of images)
 -frame_end: <int> number of the last image to be processed (-1 for no particular range of images)
 -objectID: <int> id of the object to be tracked
 -tracker_type: <string> tracker type e.g., "MEDIANFLOW", "BOOSTING", "KCF", etc.
 -error_threshold: <int> threshold (in pixels) for the force re-initialization of the tracker e.g., 6 , (-1 for no thresholding)
 -report: <File*> pointer to the report file
 Output:
 -return whether the evaluation is successfully completed (true) or not (false)
 
 Comments: auto_detect flag is not considered since we assume that all annotations are visually verified
 */
bool evaluateTracker(string test_dir_path, string file_extension, int frame_begin, int frame_end, int objectID, string tracker_type, int error_threshold, FILE *report){
    
    if(objectID == -1){
        cout<<"Evaluation could not start! Please specify one of the objects..."<<endl;
        return false;
    }
    
    //Fetch the image paths
    vector<string> image_paths;

    image_paths = scanAllFiles(test_dir_path, file_extension);
    file_extension = ".xml";
    
    int no_of_images = image_paths.size();
    if(no_of_images <= 0){
        cout<<"Number of images is <= 0..."<<endl;
        return false;
    }
    else
        cout<<"Number of images in the test directory: "<<no_of_images<<endl;
    
    Ptr<cv::Tracker> cv_tracker;
    ModifiedTrackerMedianFlow mmf_tracker;
    dlib::correlation_tracker dlib_tracker;
    KCFTracker kcf_tracker(true,false,true,true);
    
    path path_annotation;
    string path_annot = "";

    bool isOnTracking = false;
    Annotations gt, trackingResult;
    Point2d centerGT, centerTrackingResult;
    float fps;
    vector<float> fpss, fpss_init, dists, dists_x, dists_y;
    int cnt_reinit = 0;
    float dist, dist_x, dist_y;
    Mat im, im_display;
    int frame_no;
    
    fprintf(report,"Frame No,GT ROI Center (x),GT ROI Center (y),GT ROI Rect (x),GT ROI Rect (y),GT ROI Rect (width),GT ROI Rect (height), ,Dist Error,FPS,Estimated ROI Center (x),Estimated ROI Center (y),Estimated ROI Rect (x),Estimated ROI Rect (y),Estimated ROI Rect (width) \n");

    if(frame_begin == -1)
        frame_begin = 0;
    if(frame_end == -1)
        frame_end = no_of_images;
    
    int frame_counter;
    for(frame_counter = frame_begin; frame_counter < frame_end; frame_counter++){
        
        //read the image
        im = imread(test_dir_path+image_paths[frame_counter]);
        
        //get image frame no
        size_t found = image_paths[frame_counter].find_last_of("_");
        frame_no = stoi(image_paths[frame_counter].substr(found+1));
        
        //display purposes
        im_display = im.clone();
        
        //DLIB method works only with grayscale images
        if(tracker_type == "DLIB_CORRELATION" && im.channels() == 3)
            im = imread(test_dir_path+image_paths[frame_counter],IMREAD_GRAYSCALE);
        
        //display the frame
        imshow("Tracking", im_display);
        
        //read the ground truth
        path_annot = image_paths[frame_counter].substr(0, image_paths[frame_counter].size()-4);
        path_annot = test_dir_path + path_annot + ".xml";
        path path_annotation(path_annot);
        if(!exists(path_annotation)){
            cout<<"File: "<<path_annotation<<" does not exist!"<<endl;
            //fprintf(report,"%d,%d,%d,%d,%d,%d,%d, , \n",frame_no,-1,-1,-1,-1,-1,-1);
            continue;
        }
        
        //TODO: currently autodetect flag is not accurate in the annotations
        //      therefore, no particular condition applies w.r.t. this flag
        //In ideal case: if no ground truth, do not perform distance evaluation
        /*
        if(gt.object.autodetect != 0){
            cout<<"There is no ground truth for frame #"<<frame_no<<endl;
            fprintf(report,"%d,%d,%d,%d,%d,%d,%d, ,",frame_no,-1,-1,-1,-1,-1,-1);
        }*/
        
        //if there is a problem with loading the annotation, skip the file
        if(!gt.load(path_annot, objectID))
            continue;
        
        centerGT = (gt.object.bounding_box.br()+ gt.object.bounding_box.tl())*0.5;
        rectangle(im_display, gt.object.bounding_box, Scalar(0,255,0), 2, 1 );
        fprintf(report,"%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f, ,",frame_no,centerGT.x,centerGT.y,gt.object.bounding_box.x,gt.object.bounding_box.y,gt.object.bounding_box.width,gt.object.bounding_box.height);
        
        // Start timer
        double timer = (double)getTickCount();
        
        dist = -1, dist_x= -1, dist_y = -1;
        
        //Tracker initialization
        if(!isOnTracking){
            if(tracker_type == "MODIFIED_MEDIANFLOW")
                mmf_tracker.init(gt.object.bounding_box,im);
            else if(tracker_type == "GITHUB-KCF")
                kcf_tracker.init(gt.object.bounding_box,im);
            else if(tracker_type == "DLIB_CORRELATION"){
                dlib::array2d<unsigned char> dlibImageGray;
                dlib::assign_image(dlibImageGray, dlib::cv_image<unsigned char>(im));
                dlib::rectangle roi = openCVRectToDlib(gt.object.bounding_box);
                dlib_tracker.start_track(dlibImageGray, roi);
            }
            else
                cv_tracker = createInitializeOpenCVTracker(im,tracker_type,gt.object.bounding_box);
            
            isOnTracking = true;
            
            // Calculate Frames per second (FPS)
            fps = getTickFrequency() / ((double)getTickCount() - timer);
            fpss_init.push_back(fps);
            
            fprintf(report,"%d,%.2f \n",-1,fps);
        }
        //Tracking
        else{
            //copy-paste common data, i.e., file_name, image size, name etc.
            trackingResult = gt;
            
            Rect2d new_box;
            dlib::rectangle new_roi;
            double tracker_confidence = -1;
            bool is_tracking_successful = false;
            
            if(tracker_type == "MODIFIED_MEDIANFLOW")
                is_tracking_successful = mmf_tracker.update(im,new_box);
            else if(tracker_type == "GITHUB-KCF"){
                new_box = kcf_tracker.update(im);
                is_tracking_successful = true;
            }
            else if(tracker_type == "DLIB_CORRELATION"){
                dlib::array2d<unsigned char> dlibImageGray;
                dlib::assign_image(dlibImageGray, dlib::cv_image<unsigned char>(im));
                tracker_confidence = dlib_tracker.update(dlibImageGray);
                if(dlib_tracker.get_position().is_empty() == false){
                    new_roi  = dlib_tracker.get_position();
                    new_box = dlibRectangleToOpenCV(new_roi);
                    is_tracking_successful = true;
                    //cout<<new_box<<endl;
                }
                else
                    is_tracking_successful = false;
            }
            else
                is_tracking_successful = cv_tracker->update(im,new_box);
            
            fps = getTickFrequency() / ((double)getTickCount() - timer);
            fpss.push_back(fps);
            
            if(is_tracking_successful){
                //trackingResult.object.autodetect = 1;
                trackingResult.object.bounding_box = new_box;
                centerTrackingResult = (new_box.br()+ new_box.tl())*0.5;

                //calculate the MSE
                dist_x = abs(centerTrackingResult.x-centerGT.x);
                dist_y = abs(centerTrackingResult.y-centerGT.y);
                dist = norm(centerTrackingResult-centerGT);

                dists_x.push_back(dist_x);
                dists_y.push_back(dist_y);
                dists.push_back(dist);

                
                if(error_threshold != -1 && dist > error_threshold)
                    isOnTracking = false;
                
                rectangle(im_display, new_box, Scalar( 255, 0, 0 ), 2, 1 );

                fprintf(report,"%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f \n",dist,fps,centerTrackingResult.x,centerTrackingResult.y,trackingResult.object.bounding_box.x,trackingResult.object.bounding_box.y,trackingResult.object.bounding_box.width,trackingResult.object.bounding_box.height);
            }else{
                isOnTracking = false;
                fprintf(report,"%.2f,%.2f,%d,%d,%d,%d,%d,%d \n",dist,fps,-1,-1,-1,-1,-1,-1);
            }
        
        }//end of tracking

        //re-initialize the tracker
        if(!isOnTracking){
            // Tracking failure detected.
            putText(im_display, "Tracking failure detected", Point(100,80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,255),2);
            cnt_reinit++;
        }else{
            // Display tracker type on frame
            putText(im_display, tracker_type + " Tracker", Point(100,20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50),2);
        }
        
        // Display FPS on frame
        putText(im_display, "FPS : " + to_string(int(fps)), Point(100,50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50), 2);
        
        // Display frame.
        imshow("Tracking", im_display);
        
        // Exit if ESC pressed.
        int k = waitKey(1);
        if(k == 27)
            break;
    
    }

    double sum_error_x = accumulate(dists_x.begin(), dists_x.end(), 0.0);
    double mean_error_x = sum_error_x / dists_x.size();
    double sum_error_y = accumulate(dists_y.begin(), dists_y.end(), 0.0);
    double mean_error_y = sum_error_y / dists_y.size();
    
    double sum_error = accumulate(dists.begin(), dists.end(), 0.0);
    double mean_error = sum_error / dists.size();
    
    double sum_fps = accumulate(fpss.begin(), fpss.end(), 0.0);
    double mean_fps = sum_fps / fpss.size();
    
    double sum_fps_init = accumulate(fpss_init.begin(), fpss_init.end(), 0.0);
    double mean_fps_init = sum_fps_init / fpss_init.size();
    
    cout<<endl<<endl<<"No of processed frames: "<<frame_counter<<endl;
    cout<<"No of tracking re-initializations: "<<cnt_reinit<<" -> "<<(100.0*cnt_reinit)/frame_counter<<" %"<<endl;
    cout<<"Mean displacement error (in pixels): "<<mean_error<<endl;
    cout<<"Mean displacement error (on x): "<<mean_error_x<<endl;
    cout<<"Mean displacement error (on y): "<<mean_error_y<<endl;
    cout<<"Average speed (in fps): "<<mean_fps<<endl;
    
    fprintf(report,",,\n");
    fprintf(report,"No of processed frames:,%d \n",frame_counter);
    fprintf(report,"No of tracking re-initializations:,%d \n",cnt_reinit);
    fprintf(report,"Percentage of tracking re-initializations:,%.2f \n",(100.0*cnt_reinit)/frame_counter);
    fprintf(report,"Mean displacement error (in pixels):,%.2f \n",mean_error);
    fprintf(report,"Mean displacement error (on x):,%.2f \n",mean_error_x);
    fprintf(report,"Mean displacement error (on y):,%.2f \n",mean_error_y);
    fprintf(report,"Mean tracking speed (in fps):,%.2f \n",mean_fps);
    fprintf(report,"Mean initialization speed (in fps):,%.2f \n",mean_fps_init);

    return true;
    
}


int main(int argc, char **argv){
    
    string test_dir_path = argv[1];
    cout<<"Input test directory: "<<test_dir_path<<endl;

    string file_extension = argv[2]; //".jpg", ".png";
    string tracker_type = argv[3];
    cout<<"Tracker type: "<<tracker_type<<endl;

    int objectID = atoi(argv[4]);//1;
    cout<<"Object ID: "<<objectID<<endl;

    int error_threshold = atoi(argv[5]);//6;
    cout<<"Error threshold: "<<error_threshold<<endl;
    
    int frame_begin = atoi(argv[6]);//-1;
    int frame_end = atoi(argv[7]);//-1;

    char out_csv[256];

    string tracking_type = "FIXED_WINDOW";
    if(tracking_type == "FIXED_WINDOW"){

        //Rice v1 - 56 frames
        //Rect roi = Rect(Point(205,156),Point(403,632)); //w:198 x 476: Rect(Point(205,156),Point(403,632))
        //Rect roi = Rect(Point(205,256),Point(403,532)); //w:198 x 276: Rect(Point(205,256),Point(403,532))
        
        //Rice v5 - 300 frames
        //Rect roi = Rect(Point(438,184),Point(518,384)); //w:80 x 200: Rect(Point(438,184),Point(518,384))
        //Rect roi = Rect(Point(438,284),Point(518,384)); //w:80 x 100: Rect(Point(438,284),Point(518,384))
        
        //Rice v6 - 30 or 40 frames
        //Rect roi = Rect(Point(448,137),Point(531,486)); //w:83 x 350: Rect(Point(448,137),Point(531,486))
        Rect roi = Rect(Point(448,337),Point(531,486)); //w:83 x 150: Rect(Point(448,337),Point(531,486))
        
        sprintf(out_csv,"%sReport_%s_displacement_over_window%d-%d-%d-%d.csv",test_dir_path.c_str(),tracker_type.c_str(),roi.x,roi.y,roi.width,roi.height);
        FILE *report = fopen(out_csv,"w");
        if(!report)
            cout<<"Problem opening "<<out_csv<<" file!"<<endl;

        evaluateDisplacement(test_dir_path, file_extension, frame_begin, frame_end, tracker_type, objectID, roi, report);
    }
    else if(tracking_type == "MOVING_WINDOW" && tracker_type != "NCC"){
        sprintf(out_csv,"%sReport_%s_Object%d_ErrThr%d.csv",test_dir_path.c_str(),tracker_type.c_str(), objectID, error_threshold);
        
        FILE *report = fopen(out_csv,"w");
        if(!report)
            cout<<"Problem opening "<<out_csv<<" file!"<<endl;
        
        evaluateTracker(test_dir_path, file_extension, frame_begin, frame_end, objectID, tracker_type, error_threshold, report);
        
        fclose(report);
    }
    return 0;
}

/*
int main(int argc, char **argv)
{

    cout<<__cplusplus<<endl;
    #if __cplusplus==201402L
        std::cout << "C++14" << std::endl;
    #elif __cplusplus==201103L
        std::cout << "C++11" << std::endl;
    #else
        std::cout << "C++" << std::endl;
    #endif
 
 
    // List of tracker types in OpenCV 3.3
    string trackerTypes[7] = {"BOOSTING", "MIL", "KCF", "TLD","MEDIANFLOW", "GOTURN", "MODIFIED_MEDIANFLOW"};
   
    if ( argc < 2 )
    {
        printf("usage: <Video_Path> <tracker type> \n");
        return -1;
    }
    cout<<"Video path: "<<argv[1]<<endl;
    cout<<"Tracker type: "<<argv[2]<<endl;
    
    // Read video
    VideoCapture video(argv[1]);
    
    // Exit if video is not opened
    if(!video.isOpened()){
        cout << "Could not read video file!" << endl;
        return 1;
        
    }
    
    // Create a tracker
    int trackerNo = atoi(argv[2]);
    string tracker_type = trackerTypes[trackerNo];
 
    Ptr<Tracker> tracker;
    ModifiedTrackerMedianFlow MMFTracker;
    
    if(trackerNo < 6){
        if (tracker_type == "BOOSTING")
            tracker = TrackerBoosting::create();
        if (tracker_type == "MIL")
            tracker = TrackerMIL::create();
        if (tracker_type == "KCF")
            tracker = TrackerKCF::create();
        if (tracker_type == "TLD")
            tracker = TrackerTLD::create();
        if (tracker_type == "MEDIANFLOW")
            tracker = TrackerMedianFlow::create();
        if (tracker_type == "GOTURN")
            tracker = TrackerGOTURN::create();
    }
    
    // Read first frame
    Mat frameColor;
    if(!video.read(frameColor)){
        cout<<"Problem reading the first frame from the video..."<<endl;
        return 0;
    }
    
    // Create a new matrix to hold the gray image
    Mat frame;
    // convert RGB image to gray
    cvtColor(frameColor, frame, CV_BGR2GRAY);
    //resize(frame,frame,cv::Size(),0.5,0.5);
    
    //TODO: read it from the ground truth annotations
    // Define initial bounding box
    Rect2d bbox(470, 130, 50, 50);
    
    // Uncomment the line below to select a different bounding box using GUI
    //bbox = selectROI(frame, false);
    
    // Display bounding box.
    rectangle(frame, bbox, Scalar( 255, 0, 0 ), 2, 1 );
    
    imshow("Tracking", frame);
    
    if(trackerNo < 6)
        tracker->init(frame, bbox);
    else
        MMFTracker.init(bbox,frame);
    
    int ind = 1;
    while(video.read(frameColor))
    {
        cvtColor(frameColor, frame, CV_BGR2GRAY);
        //resize(frame,frame,cv::Size(),0.5,0.5);

        // Start timer
        double timer = (double)getTickCount();
        
        //cout<<ind<<"\t old:"<<bbox<<endl;
        
        // Update the tracking result
        bool ok;
        if(trackerNo < 6)
            ok = tracker->update(frame, bbox);
        else
            ok = MMFTracker.update(frame, bbox);
        
        //cout<<"new:"<<bbox<<endl<<endl<<endl;
        
        // Calculate Frames per second (FPS)
        float fps = getTickFrequency() / ((double)getTickCount() - timer);
        
        if (ok)
        {
            // Tracking success : Draw the tracked object
            rectangle(frame, bbox, Scalar( 255, 0, 0 ), 2, 1 );
        }
        else
        {
            // Tracking failure detected.
            putText(frame, "Tracking failure detected", Point(100,80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,255),2);
        }
        
        // Display tracker type on frame
        putText(frame, tracker_type + " Tracker", Point(100,20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50),2);
        
        // Display FPS on frame
        putText(frame, "FPS : " + to_string(int(fps)), Point(100,50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50), 2);
        
        // Display frame.
        imshow("Tracking", frame);
        ind++;
        
        // Exit if ESC pressed.
        int k = waitKey(1);
        if(k == 27)
        {
            break;
        }
        
    }
}
 */
