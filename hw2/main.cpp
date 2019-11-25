#include <iostream>
#include <opencv2/opencv.hpp>
#include <string.h>

using namespace cv;
using namespace std;

int main(int argc, char** argv ) {
    string imageFilePath;
    try
    {
        imageFilePath = argv[1];
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    
    Mat Img = imread(imageFilePath);
    // 导入图像转为灰度图
    Mat grayscaleImg = imread(imageFilePath, IMREAD_GRAYSCALE);

    // 测试一下灰度图
    vector<int> compression_params;
    compression_params.push_back(IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(9);
    // imwrite(imageFilePath+"-grayscale.png",grayscaleImg,compression_params);

    // 二值化
    Mat binaryImg;
    double thresh = 128;
    threshold(grayscaleImg, binaryImg, thresh, 255, THRESH_BINARY);
    imwrite(imageFilePath+"-binary-"+to_string((int)thresh)+".png",binaryImg,compression_params);

    // 提取轮廓
    vector< vector<Point> > outline;
    findContours(binaryImg, outline, RETR_TREE, CHAIN_APPROX_SIMPLE);
    
    // 椭圆拟合
    for(auto &item : outline) {
        RotatedRect res;
        if(item.size() >= 20) {
            res = fitEllipse(item);
            ellipse(Img,res,Scalar(200,200,250),4,LINE_AA);
        }
    }

    // output image
    imwrite(imageFilePath+"-result.png",Img,compression_params);

}