// opencv version 4.1.1
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string.h>

using namespace cv;
using namespace std;

int main(int argc, char** argv ) {
    string datadir = "./assets";
    if(argc>1)
    {
        datadir = argv[1];
    }
    int frameWidth = 960;
    int frameHeight = 540;
    string name = "processedVideo.avi";
    int fps = 30;
    Size size = Size(frameWidth, frameHeight);
    VideoWriter writer = VideoWriter(name, VideoWriter::fourcc('D', 'I', 'V', 'X'),fps,size,true);

    vector<string> imgpaths;
    vector<string> avipaths;
    glob(datadir+"/*.jpg",imgpaths,false);
    glob(datadir+"/*.avi",avipaths,false);
    // test
    cout << "import file:" << endl;
    for (auto &item : imgpaths)
    {
        cout << item << endl;
    }
    cout << avipaths[0] << endl;
    // test ok
    string myinfo = "Author: wangjinkai 3170102728";
    // image
    for (auto &item : imgpaths)
    {
        Mat img;
        try
        {
            img = imread(item);
            if (!img.data)
            {
                throw 0;
            }
        }
        catch(int)
        {
            cerr << "image not valid!" << endl;
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
        }
        Mat imgP;
        resize(img, imgP, size);
        putText(
            imgP, 
            myinfo, 
            cv::Point(250,500), 
            cv::FONT_HERSHEY_SCRIPT_COMPLEX, 
            1, 
            cv::Scalar(200,200,250), 
            2,
            LINE_AA,
            false
        );
        for(int j=0; j<60; j++) {
            writer.write(imgP);
        }
    }
    // video
    string videoPath = avipaths[0];
    VideoCapture capture = VideoCapture(videoPath);
    Mat frame,framep;
    while(1)
    {
        capture >> frame;
        if(frame.empty())
            break;
        resize(frame, framep, size);
        putText(
            framep, 
            myinfo, 
            cv::Point(250,500), 
            cv::FONT_HERSHEY_SCRIPT_COMPLEX, 
            1, 
            cv::Scalar(200,200,250), 
            2,
            LINE_AA,
            false
        );
        writer.write(framep);
    }
    writer.release();
    cout << "Process successfully!" << endl;
}