//
//  main.cpp
//  ObjectTracking
//
//  Created by Nuri Murat ARAR on 09.10.17.
//  Copyright © 2017 Nuri Murat ARAR. All rights reserved.
//

#include <iostream>
#include <stdio.h>

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>

using namespace cv;
using namespace std;

// Convert to string
#define SSTR( x ) static_cast< std::ostringstream & >( \
( std::ostringstream() << std::dec << x ) ).str()

int main(int argc, char **argv)
{
    // List of tracker types in OpenCV 3.3
    string trackerTypes[6] = {"BOOSTING", "MIL", "KCF", "TLD","MEDIANFLOW", "GOTURN"};
   
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
    if(!video.isOpened())
    {
        cout << "Could not read video file!" << endl;
        return 1;
        
    }
    
    // Create a tracker
    string trackerType = trackerTypes[atoi(argv[2])];
    
    Ptr<Tracker> tracker;
    
    if (trackerType == "BOOSTING")
        tracker = TrackerBoosting::create();
    if (trackerType == "MIL")
        tracker = TrackerMIL::create();
    if (trackerType == "KCF")
        tracker = TrackerKCF::create();
    if (trackerType == "TLD")
        tracker = TrackerTLD::create();
    if (trackerType == "MEDIANFLOW")
        tracker = TrackerMedianFlow::create();
    if (trackerType == "GOTURN")
        tracker = TrackerGOTURN::create();
    
    // Read first frame
    Mat frame;
    bool ok = video.read(frame);
    
    // Define initial boundibg box
    Rect2d bbox(287, 23, 86, 320);
    
    // Uncomment the line below to select a different bounding box
    bbox = selectROI(frame, false);
    
    // Display bounding box.
    rectangle(frame, bbox, Scalar( 255, 0, 0 ), 2, 1 );
    imshow("Tracking", frame);
    
    tracker->init(frame, bbox);
    
    while(video.read(frame))
    {
        // Start timer
        double timer = (double)getTickCount();
        
        // Update the tracking result
        bool ok = tracker->update(frame, bbox);
        
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
        putText(frame, trackerType + " Tracker", Point(100,20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50),2);
        
        // Display FPS on frame
        putText(frame, "FPS : " + SSTR(int(fps)), Point(100,50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50), 2);
        
        // Display frame.
        imshow("Tracking", frame);
        
        // Exit if ESC pressed.
        int k = waitKey(1);
        if(k == 27)
        {
            break;
        }
        
    }
}

/*int main(int argc, const char * argv[]) {
 // insert code here...
 std::cout << "Hello, World!\n";
 std::cout << "OpenCV version "<<CV_VERSION<<std::endl;
 
 if ( argc != 2 )
 {
 printf("usage: DisplayImage.out <Image_Path>\n");
 return -1;
 }
 cout<<argv[1]<<endl;
 Mat image;
 image = imread( argv[1], 1 );
 if ( !image.data )
 {
 printf("No image data \n");
 return -1;
 }
 namedWindow("Display Image", WINDOW_AUTOSIZE );
 imshow("Display Image", image);
 waitKey(0);
 return 0;
 }*/
