#include <opencv2/opencv.hpp>
#include <iostream>
#include <string.h>
#include <vector>

using namespace cv;
using namespace std;

Size standardSize = Size(128,128);
int height = 128;
int width = 128;
string modelName = "model.yml";
string testfile = "";

int main(int argc, char** argv ) {
    // read params
    if(argc!=3) {
        cout<< "参数不够" <<endl;
        return 0;
    }else {
        testfile = argv[1];
        modelName = argv[2];
    }

    // read model
    Mat eigenM = Mat();
    FileStorage storage("../"+modelName, FileStorage::READ);
    storage["model"] >> eigenM;
    storage.release();
    cout<<"eigenM Size:"<<endl;
    cout<<"行数: "<<eigenM.rows<<endl; // should be eigenCount
    cout<<"列数: "<<eigenM.cols<<endl; // should be height*width
    cout<<"eigenM matrix done\n"; 
    cout<<endl;

    // 测试
    cout<<"--------Test EigenFace Model---------"<<endl;
    // 求一下特征矩阵的转置
    int eigenCount = eigenM.rows;
    Mat eigenMT = Mat(height*width,eigenCount,CV_32FC1);
    eigenMT = eigenM.t();
    // read test face
    string testDatadir = "../JAFFE/test/";
    Mat testImg = imread(testDatadir+testfile, IMREAD_GRAYSCALE); // single channel
    resize(testImg, testImg, standardSize);
    imshow("testFace",testImg);
    // normalize(testImg, testImg, 255, 0, NORM_MINMAX);
    // 构建testFace的向量testMat
    Mat testMat = testImg.reshape(1,1);
    testMat.convertTo(testMat,CV_32FC1);
    cout<<"testMat Size:"<<endl;
    cout<<"行数: "<<testMat.rows<<endl; // should be 1
    cout<<"列数: "<<testMat.cols<<endl; // should be height*width
    cout<<"testMat matrix done\n";
    // 计算testMat在特征空间的向量
    Mat testVector = testMat * eigenMT;
    cout<<"testVector Size:"<<endl;
    cout<<"行数: "<<testVector.rows<<endl; // should be 1
    cout<<"列数: "<<testVector.cols<<endl; // should be eigenCount
    cout<<"testVector matrix done\n";

    // 读入可查找face
    string trainDatadir = "../JAFFE/train";
    vector<string> imgpaths;
    glob(trainDatadir+"/*.tiff",imgpaths,false);
    int k=imgpaths.size();
    // 遍历寻找和他相似的图片
    Mat judgeMat = Mat(1,height*width,CV_32FC1);
    Mat finalMat = Mat(1,height*width,CV_32FC1);
    string finalFileName = "";
    double minDistance=1e30;
    for(int i=0;i<k;i++) {
        string item = imgpaths[i];
        Mat img = imread(item, IMREAD_GRAYSCALE); // single channel
        resize(img, img, standardSize);
        Mat picVector = img.reshape(1,1).clone();
        picVector.convertTo(picVector,CV_32FC1);

        judgeMat = picVector.clone();
        Mat judgeVector = judgeMat * eigenMT;
        if(i==0){
            cout<<"judgeVector Size:"<<endl;
            cout<<"行数: "<<judgeVector.rows<<endl; // should be 1
            cout<<"列数: "<<judgeVector.cols<<endl; // should be eigenCount
            cout<<"judgeVector matrix done\n";
        }
        Mat diffVector = testVector-judgeVector;
        double distance = norm(diffVector, NORM_L2);
        cout << "distance of image "<<imgpaths[i]<<": "<<distance<<endl;
        if(distance<minDistance) {
            minDistance = distance;
            finalMat = judgeMat.clone();
            finalFileName = item;
        }
    }
    cout << "test done!\n";
    cout << "test pic: "<< testDatadir<<testfile << endl;
    cout << "find pic: "<< finalFileName << endl;
    Mat resMat = finalMat.reshape(1,height);
    resMat.convertTo(resMat,CV_8UC1);
    imshow("resFace",resMat);

    // rebuildFace
    Mat meanFace = imread("../mytrain/meanFace.png", IMREAD_GRAYSCALE);
    cout<<"meanFace Size:"<<endl;
    cout<<"行数: "<<meanFace.rows<<endl; // should be height
    cout<<"列数: "<<meanFace.cols<<endl; // should be width
    cout<<"meanFace matrix done\n";
    meanFace.convertTo(meanFace,CV_32FC1);
    normalize(meanFace, meanFace, 255, 0, NORM_MINMAX);
    Mat rebuildMat = testVector*eigenM;
    normalize(rebuildMat, rebuildMat, 255, 0, NORM_MINMAX);
    rebuildMat = meanFace.reshape(1,1) + rebuildMat;
    normalize(rebuildMat, rebuildMat, 255, 0, NORM_MINMAX);
    rebuildMat = rebuildMat.reshape(1,height);
    rebuildMat.convertTo(rebuildMat,CV_8UC1);
    imshow("rebuildFace",rebuildMat);

    waitKey(0);
}

// test file: KM.SA5.13.tiff