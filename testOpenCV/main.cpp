#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main(int argc, char** argv ) {
    Mat picture = imread("./test.jpg");
    //图片必须添加到工程目下
    //也就是和test.cpp文件放在一个文件夹下！！！
    imshow("测试程序", picture);
    waitKey(20000);
}