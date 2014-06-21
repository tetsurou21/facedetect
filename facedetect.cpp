/*
   1. 顔１目２が見つかる画像のみを対象とする

   2. 入力されたデータと、全データを照合する
   　 このとき、画像特徴量の整合を、いろいろな手法でスコア計算する

   　・エリア特徴量
   　  　面ベース、ウィンドウ間での特徴量の計算を行う系
   　　　１．ZNCC、ZSSD、POC
   　　　２．ヒストグラム

 　　・局所特徴量
   　　点ベース、ポイント間の整合性を見る
   　　　１．FAST、SURF、SIFT
*/


#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <time.h>
#include <opencv2/nonfree/nonfree.hpp>
#include <iostream>

using namespace std;
using namespace cv;

vector<Rect> findFaces(Mat parent, string cascadeFile) {
  CascadeClassifier classifiler;
  if(!classifiler.load(cascadeFile)) {
    cerr << "failed to load " << cascadeFile << endl;
    return vector<Rect>();
  }
  Mat imgGray;
  cvtColor(parent, imgGray, CV_BGR2GRAY);
  equalizeHist(imgGray, imgGray);

  vector<Rect> faceRects;

  classifiler.detectMultiScale(imgGray, faceRects,
      1.1, 2,
      CV_HAAR_SCALE_IMAGE, 
      Size(30,30));
  return faceRects;

}

int main(int argc, char *argv[])
{
  srand(time(NULL));

  if (argc != 3) {
    fprintf(stderr, "Usage: %s from to\n", argv[0]);
    return -1;
  }
  string srcFile = argv[1];
  string targetFile = argv[2];

  Mat srcImage = imread(srcFile, 1);
  if (srcImage.empty()) {
    cerr << "failed to load " << srcFile << endl;
    return -1;
  }
  Mat targetImage = imread(targetFile, 1);
  if (targetImage.empty()) {
    cerr << "failed to load " << targetFile << endl;
    return -1;
  }

  vector<Rect> srcRects = findFaces(srcImage, "./haarcascade_frontalface_alt.xml");
  Rect srcRect = srcRects.at(0);
  Mat srcFaceImage = srcImage(srcRect);
  vector<Rect> targetRects = findFaces(targetImage, "./lbpcascade_animeface.xml");

  vector<Rect>::const_iterator r = targetRects.begin();
  for(; r != targetRects.end(); ++r) {
    Rect faceRect = *r;
    Mat targetFaceImage = targetImage(faceRect);
    Mat resizedSrcFaceImage(faceRect.width, faceRect.height, targetFaceImage.channels());
    resize(srcFaceImage, resizedSrcFaceImage, resizedSrcFaceImage.size());

    // overwrite pixels
    for (int i = 0; i < faceRect.width; ++i) {
      for (int j = 0; j < faceRect.height; ++j) {
        targetFaceImage.at<Vec3b>(i, j) = resizedSrcFaceImage.at<Vec3b>(i, j);
      }
    }    
  }
  imwrite("detected.jpg", targetImage);

  namedWindow("detect", CV_WINDOW_AUTOSIZE|CV_WINDOW_FREERATIO);
  imshow("detect", targetImage);
  waitKey(0);
  return 0;
}
