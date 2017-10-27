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
    Object object;
    void load(const string &filename);
    void save(const string &filename);
};

// Loads annotations structure from the specified XML file
void Annotations::load(const string &filename){
    // Create an empty property tree object
    using boost::property_tree::ptree;
    ptree pt;
    
    // Load the XML file into the property tree. If reading fails
    // (cannot open file, parse error), an exception is thrown.
    read_xml(filename, pt);
    
    folder_name = pt.get<string>("annotation.folder");
    file_name = pt.get<string>("annotation.filename");
    path = pt.get<string>("annotation.path");
    
    image_size.width = pt.get<int>("annotation.size.width");
    image_size.height = pt.get<int>("annotation.size.height");
    image_size.depth = pt.get<int>("annotation.size.depth");

    //segmented = pt.get<int>("annotation.segmented");
    object.name = pt.get<string>("annotation.object.name");
    object.pose = pt.get<string>("annotation.object.pose");
    object.truncated = pt.get<int>("annotation.object.truncated");
    object.difficult = pt.get<int>("annotation.object.difficult");
    object.id = pt.get<int>("annotation.object.id");
    object.autodetect = pt.get<int>("annotation.object.autodetect");
    
    Point2d p1(pt.get<double>("annotation.object.bndbox.xmin"),pt.get<double>("annotation.object.bndbox.ymin"));
    Point2d p2(pt.get<double>("annotation.object.bndbox.xmax"),pt.get<double>("annotation.object.bndbox.ymax"));
    object.bounding_box = Rect(p1, p2);

   /*
    // Iterate over the debug.modules section and store all found
    // modules in the m_modules set. The get_child() function
    // returns a reference to the child at the specified path; if
    // there is no such child, it throws. Property tree iterators
    // are models of BidirectionalIterator.
    BOOST_FOREACH(ptree::value_type &v,
                  pt.get_child("debug.modules"))
    m_modules.insert(v.second.data());
    */
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

Ptr<Tracker> createInitializeOpenCVTracker(Mat &im, string tracker_type, Rect boundingBox){
    Ptr<Tracker> tracker;

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
    
    tracker->init(im, boundingBox);
    
    return tracker;
}

bool evaluateTracker(string test_dir_path, int objectID, string tracker_type, int errorThreshold, FILE *report){
    
    if(objectID == -1){
        cout<<"Evaluation could not start! Please specify one of the objects..."<<endl;
        return false;
    }
    
    //Fetch the images and ground truth files
    vector<string> image_paths, gt_paths;
    string file_extension = ".jpg";
    //string file_extension = ".png";
    image_paths = scanAllFiles(test_dir_path, file_extension);
    file_extension = ".xml";
    gt_paths = scanAllFiles(test_dir_path, file_extension);
    
    int no_of_images = image_paths.size();
    if(no_of_images <= 0){
        cout<<"Number of images is <= 0..."<<endl;
        return false;
    }
    else
        cout<<"Number of images in the test directory: "<<no_of_images<<endl;
    
    int no_of_gtfiles = gt_paths.size();
    if(no_of_gtfiles <= 0){
        cout<<"Number of ground truth files is <= 0..."<<endl;
        return false;
    }
    else
        cout<<"Number of ground truth files in the test directory: "<<no_of_gtfiles<<endl;
    if(no_of_images != no_of_gtfiles){
        cout<<"Mismatch in the number of ground truth files and images..."<<endl;
        return false;
    }
    
    Ptr<Tracker> cv_tracker;
    ModifiedTrackerMedianFlow mmf_tracker;
    dlib::correlation_tracker dlib_tracker;

    bool isOnTracking = false;
    
    Annotations gt, trackingResult;
    Point2d centerGT, centerTrackingResult;
    float fps;
    vector<float> fpss, fpss_init, dists;
    int cnt_reinit = 0;
    float dist;
    Mat im, im_display;
    
    fprintf(report,"Frame No,GT ROI Center (x),GT ROI Center (y),GT ROI Rect (x),GT ROI Rect (y),GT ROI Rect (width),GT ROI Rect (height), ,Dist Error,FPS,Estimated ROI Center (x),Estimated ROI Center (y),Estimated ROI Rect (x),Estimated ROI Rect (y),Estimated ROI Rect (width) \n");


    for(int i = 0; i < no_of_images; i++){
        
        //read the image
        im = imread(test_dir_path+image_paths[i]);
        //display purposes
        im_display = im.clone();
        
        //DLIB method works only with grayscale images
        if(tracker_type == "DLIB_CORRELATION" && im.channels() == 3)
            im = imread(test_dir_path+image_paths[i],IMREAD_GRAYSCALE);
        
        //display the frame
        imshow("Tracking", im_display);
        
        //read the ground truth
        gt.load(test_dir_path+gt_paths[i]);
        centerGT = (gt.object.bounding_box.br()+ gt.object.bounding_box.tl())*0.5;

        rectangle(im_display, gt.object.bounding_box, Scalar(0,255,0), 2, 1 );
        
        fprintf(report,"%d,%2f,%2f,%2f,%.2f,%.2f,%.2f, ,",i,centerGT.x,centerGT.y,gt.object.bounding_box.x,gt.object.bounding_box.y,gt.object.bounding_box.width,gt.object.bounding_box.height);
        
        // Start timer
        double timer = (double)getTickCount();
        
        //Tracker initialization
        if(!isOnTracking){
            if(tracker_type == "MODIFIED_MEDIANFLOW")
                mmf_tracker.init(gt.object.bounding_box,im);
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
            
            fprintf(report,"%d,%2f \n",0,fps);

        }
        //Tracking
        else{
            //copy-paste common data, i.e., file_name, image size, name etc.
            trackingResult = gt;
            Rect2d new_box;
            dlib::rectangle new_roi;
            double tracker_confidence = -1;
            
            if(tracker_type == "MODIFIED_MEDIANFLOW")
                mmf_tracker.update(im,new_box);
            else if(tracker_type == "DLIB_CORRELATION"){
                dlib::array2d<unsigned char> dlibImageGray;
                dlib::assign_image(dlibImageGray, dlib::cv_image<unsigned char>(im));
                tracker_confidence = dlib_tracker.update(dlibImageGray);
                new_roi  = dlib_tracker.get_position();
                new_box = dlibRectangleToOpenCV(new_roi);
            }
            else
                cv_tracker->update(im,new_box);
            
            fps = getTickFrequency() / ((double)getTickCount() - timer);
            fpss.push_back(fps);

            trackingResult.object.autodetect = 1;
            trackingResult.object.bounding_box = new_box;
            centerTrackingResult = (new_box.br()+ new_box.tl())*0.5;

            //calculate the MSE
            dist = norm(centerTrackingResult-centerGT);
            dists.push_back(dist);

            fprintf(report,"%2f,%2f,%2f,%2f,%2f,%.2f,%.2f,%.2f \n",dist,fps,centerTrackingResult.x,centerTrackingResult.y,trackingResult.object.bounding_box.x,trackingResult.object.bounding_box.y,trackingResult.object.bounding_box.width,trackingResult.object.bounding_box.height);

            if(dist > errorThreshold){
                // Tracking failure detected.
                putText(im_display, "Tracking failure detected", Point(100,80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,255),2);
                
                //re-initialize the tracker
                isOnTracking = false;
                cnt_reinit++;
            }
            else{
                rectangle(im_display, new_box, Scalar( 255, 0, 0 ), 2, 1 );
            }
            
        }//end of tracking


        // Display tracker type on frame
        putText(im_display, tracker_type + " Tracker", Point(100,20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50),2);
        
        // Display FPS on frame
        putText(im_display, "FPS : " + to_string(int(fps)), Point(100,50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50), 2);
        
        // Display frame.
        imshow("Tracking", im_display);
        
        // Exit if ESC pressed.
        int k = waitKey(1);
        if(k == 27)
        {
            break;
        }
        
    }
    
    double sum_error = accumulate(dists.begin(), dists.end(), 0.0);
    double mean_error = sum_error / dists.size();
    
    double sum_fps = accumulate(fpss.begin(), fpss.end(), 0.0);
    double mean_fps = sum_fps / fpss.size();
    
    double sum_fps_init = accumulate(fpss_init.begin(), fpss_init.end(), 0.0);
    double mean_fps_init = sum_fps_init / fpss_init.size();
    
    cout<<endl<<endl<<"No of processed frames: "<<no_of_images<<endl;
    cout<<"No of tracking re-initializations: "<<cnt_reinit<<" -> "<<(100.0*cnt_reinit)/no_of_images<<" %"<<endl;
    cout<<"Average displacement error (in pixels): "<<mean_error<<endl;
    cout<<"Average speed (in fps): "<<mean_fps<<endl;
    
    fprintf(report,"\n");
    fprintf(report,"No of processed frames:,%d \n",no_of_images);
    fprintf(report,"No of tracking re-initializations:,%d \n",cnt_reinit);
    fprintf(report,"Percentage of tracking re-initializations:,%.2f \n",(100.0*cnt_reinit)/no_of_images);
    fprintf(report,"Average displacement error (in pixels):,%.2f \n",mean_error);
    fprintf(report,"Average tracking speed (in fps):,%.2f \n",mean_fps);
    fprintf(report,"Average initialization speed (in fps):,%.2f \n",mean_fps_init);

    return true;
    
}

int main(int argc, char **argv)
{
    //string test_dir_path = "/Users/nma/Documents/VOANN/data/rice_v1/rice1_orig/";
    //string tracker_type = "BOOSTING"; //working only with RGB images
    //string tracker_type = "KCF"; //working only with RGB images
    //string tracker_type = "TLD"; //working only with RGB images
    //string tracker_type = "MIL"; //working with both RGB and Grayscale images
    //string tracker_type = "MEDIANFLOW"; //working with both RGB and Grayscale images
    //string tracker_type = "MODIFIED_MEDIANFLOW"; //working with both RGB and Grayscale images
    //string tracker_type = "DLIB_CORRELATION"; //working only with Grayscale images
    //string tracker_type = "GOTURN"; //results are bad!!!
    
    string test_dir_path = argv[1];
    string tracker_type = argv[2];
    int errorThreshold = atoi(argv[4]);//6;
    int objectID = atoi(argv[3]);//1;
    cout<<"Input test directory: "<<test_dir_path<<endl;
    cout<<"Tracker type: "<<tracker_type<<endl;
    cout<<"Error threshold: "<<errorThreshold<<endl;
    
    char out_csv[256];
    sprintf(out_csv,"%sReport_%s_Object-%d_ErrThr-%d.csv",test_dir_path.c_str(),tracker_type.c_str(), objectID, errorThreshold);
    
    FILE *report = fopen(out_csv,"w");
    if(!report)
        cout<<"Problem opening "<<out_csv<<" file!"<<endl;
    
    evaluateTracker(test_dir_path, objectID, tracker_type, errorThreshold, report);
    
    fclose(report);
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
