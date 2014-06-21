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

int saveRoi(Mat faceMat, string cascadeFile, string outputFile) {
  CascadeClassifier classifiler;
  if(!classifiler.load(cascadeFile)) {
    cerr << "failed to load " << cascadeFile << endl;
    return -1;
  }

  vector<Rect> roiRects;

  classifiler.detectMultiScale(faceMat, roiRects,
      1.1, 3,
      CV_HAAR_SCALE_IMAGE, 
      Size(10,10));

  if (roiRects.size() != 1) {
    return -1;
  }
  Rect roiRect = roiRects.at(0);
  Mat roi = faceMat(roiRect);
  imwrite(outputFile, roi);
  return 0;
}

int main(int argc, char *argv[])
{
  srand(time(NULL));

  if (argc != 3) {
    fprintf(stderr, "Usage: %s file dir\n", argv[0]);
    return -1;
  }

  string imagename = argv[1];
  string dirname = argv[2];

  Mat img = imread(imagename, 1);
  if(img.empty()) {
    cerr << imagename << " is empty" << endl;
    return -1;
  }

  double scale = img.rows / 200.0; // ★横の解像度200pixelsにする
  Mat gray, smallImg(saturate_cast<int>(img.rows/scale), saturate_cast<int>(img.cols/scale), CV_8UC1);

  // グレースケール画像に変換
  cvtColor(img, gray, CV_BGR2GRAY);
  // 処理時間短縮のために画像を縮小
  resize(gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR);
  equalizeHist( smallImg, smallImg);

  // 分類器の読み込み
  string cascadeName = "./haarcascade_frontalface_alt.xml"; // Haar-like
  CascadeClassifier cascade;
  if(!cascade.load(cascadeName)) {
    cerr << "failed to load " << cascadeName << endl;
    return -1;
  }

  vector<Rect> faceRects;
  /// マルチスケール（顔）探索xo
  // 画像，出力矩形，縮小スケール，最低矩形数，（フラグ），最小矩形
  cascade.detectMultiScale(smallImg, faceRects,
      1.1, 2,
      CV_HAAR_SCALE_IMAGE,
      Size(30, 30));

  if (faceRects.size() != 1) {
    cerr << "faces are too many or too few faces: " << faceRects.size() << endl;
    return -1;
  }

  Rect faceRect = faceRects.at(0);

  // 眼の検出
  string nested_cascadeName = "./haarcascade_eye.xml";
  CascadeClassifier nested_cascade;
  if(!nested_cascade.load(nested_cascadeName)) {
    cerr << "failed to load " << nested_cascadeName << endl;
    return -1;
  }

  int nr_count = 0;
  Mat faceMat;
  vector<Rect> nestedObjects;

  faceMat = smallImg(faceRect);
  /// マルチスケール（目）探索
  // 画像，出力矩形，縮小スケール，最低矩形数，（フラグ），最小矩形
  nested_cascade.detectMultiScale(faceMat, nestedObjects,
      1.1, 3,
      CV_HAAR_SCALE_IMAGE, 
      Size(10,10));

  if (nestedObjects.size() != 2) {
    cerr << "eyes are too many or too few: " << nestedObjects.size() << endl;
    return -1;
  }

  // 画像を保存する
  // 顔領域のみを保存 ROIで
  imwrite(dirname + "/face.jpg", faceMat);

  Rect eyeRect1 = nestedObjects.at(0);
  Rect eyeRect2 = nestedObjects.at(1);
  Mat left_eye, right_eye;

  if (eyeRect1.x < eyeRect2.x) {
    left_eye = faceMat(eyeRect2);
    right_eye = faceMat(eyeRect1);
  }
  else {
    left_eye = faceMat(eyeRect1);
    right_eye = faceMat(eyeRect2);
  }
  imwrite(dirname + "/left_eye.jpg",left_eye);
  imwrite(dirname + "/right_eye.jpg", right_eye);

  saveRoi(faceMat, "./haarcascade_mcs_nose.xml", dirname + "/nose.jpg");
  saveRoi(faceMat, "./haarcascade_mcs_mouth.xml", dirname + "/mouth.jpg");
}
