#include <opencv2/opencv.hpp>
#include <iostream>
#include <string.h>
#include <vector>

using namespace cv;
using namespace std;

Size standardSize = Size(128,128);
int k = 40; // 默认40张人脸图
double energyPercent = 0.5;
string modelName = "model.yml";

int main(int argc, char** argv ) {
    // read params
    if(argc!=3) {
        cout<< "参数不够" <<endl;
        return 0;
    }else {
        energyPercent = atof(argv[1]);
        modelName = argv[2];
    }
    // compression_params
    vector<int> compression_params;
    compression_params.push_back(IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(9);

    // read faces
    vector<Mat> faceMats; // 存放所有face
    cout<<"--------READING DATA---------"<<endl;
    string trainDatadir = "../JAFFE/train";
    vector<string> imgpaths;
    glob(trainDatadir+"/*.tiff",imgpaths,false);
    k=imgpaths.size();
    cout<<"img number: "<<k<<endl;
    for (int i=0; i<k; i++)
    {
        string item = imgpaths[i];
        // cout<<"Read the Photo: "<<item<<endl;
        Mat img = imread(item, IMREAD_GRAYSCALE); // single channel
        // resize
        resize(img, img, standardSize);
        faceMats.push_back(img);
    }
    cout<<endl;

    int height = faceMats[0].rows;
    int width = faceMats[0].cols;

    // 计算均值矩阵
    cout<<"--------CALC Mean Matrix---------"<<endl;
    Mat allPicVector = Mat::zeros(k,height*width,CV_32FC1);
    for (int i = 0; i < k; i++)
    {
        Mat picVector = faceMats[i].reshape(1,1).clone();
        // normalize
        // normalize(picVector, picVector, 255, 0, NORM_MINMAX);
        picVector.convertTo(picVector,CV_32FC1);
        if(i==0)
        {
            cout<<"picVector Size:"<<endl;
            cout<<"行数: "<<picVector.rows<<endl; // should be 1
            cout<<"列数: "<<picVector.cols<<endl; // should be height*width
            // cout<<picVector<<endl;
        }
        picVector.copyTo(allPicVector.row(i));
    }
    cout<<"allPicVector Size:"<<endl;
    cout<<"行数: "<<allPicVector.rows<<endl; // should be k
    cout<<"列数: "<<allPicVector.cols<<endl; // should be height*width
    // cout<<allPicVector.row(k-1)<<endl;

    // 计算均值图
    Mat meanMatT = Mat(1,height*width,CV_32FC1);
    for (int i = 0; i < height*width; i++)
    {
        meanMatT.col(i) = sum(allPicVector.col(i)) / allPicVector.rows; 
    }
    cout<<"meanMatT Size:"<<endl;
    cout<<"行数: "<<meanMatT.rows<<endl; // should be 1
    cout<<"列数: "<<meanMatT.cols<<endl; // should be height*width
    Mat meanMat = meanMatT.reshape(1,height);
    meanMat.convertTo(meanMat,CV_8UC1);
    cout<<"meanMat Size:"<<endl;
    cout<<"行数: "<<meanMat.rows<<endl; // should be height
    cout<<"列数: "<<meanMat.cols<<endl; // should be width
   
    imwrite("meanFace.png", meanMat, compression_params);
    cout<<"Output Mean Face done!"<<endl;
    // imshow("MeanFace",meanMat);
    // waitKey(0);
    cout<<endl;

    // 计算协方差矩阵
    for(int i=0;i<allPicVector.rows;i++)
    {
        Mat diffMat = allPicVector.row(i) - meanMatT;
        // if(i==0) cout<<diffMat<<endl;
        diffMat.copyTo(allPicVector.row(i));
    }
    Mat covMat=Mat();
    covMat = allPicVector * allPicVector.t();
    cout<<"covMat Size:"<<endl;
    cout<<"行数: "<<covMat.rows<<endl; // should be height*width
    cout<<"列数: "<<covMat.cols<<endl; // should be height*width
    cout<<"cov matrix done\n";
    
    Mat eigenVectors = Mat();
    Mat eigenValues = Mat();
    eigen(covMat,eigenValues,eigenVectors);
    cout<<"original eigen count: "<<eigenVectors.rows<<endl;
    int eigenCount = eigenVectors.rows*energyPercent;
    cout<<"eigen count: "<<eigenCount<<endl;

    Mat eigenM = Mat(eigenCount,height*width,CV_32FC1);
    // 构建映射空间
    Mat eigenFace10 = Mat(10,height*width,CV_32FC1);
    for(int i=0; i<eigenCount; i++){
        Mat eigenFace = Mat();
        eigenFace = allPicVector.t() * eigenVectors.col(i);
        eigenFace = eigenFace.t();
        // 保存前10个特征脸为一张脸
        if(i<10) {
            eigenFace.copyTo(eigenFace10.row(i));
        }
        eigenFace.copyTo(eigenM.row(i));
        eigenFace = eigenFace.reshape(1,height);
    }
    // 保存前10个特征脸为一张脸
    eigenFace10 = eigenFace10.reshape(1,10*height);
    imwrite("eigenFace10.png", eigenFace10, compression_params);

    cout<<"eigenM Size:"<<endl;
    cout<<"行数: "<<eigenM.rows<<endl; // should be eigenCount
    cout<<"列数: "<<eigenM.cols<<endl; // should be height*width
    cout<<"eigenM matrix done\n"; 
    cout<<endl;
    // save model
    FileStorage storage("../"+modelName, FileStorage::WRITE);
    storage << "model" << eigenM;
    storage.release();
}