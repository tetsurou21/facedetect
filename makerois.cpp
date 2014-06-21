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

int main(int argc, char *argv[])
{
  srand(time(NULL));

  char imagename[512];

  if (argc != 3) {
    fprintf(stderr, "Usage: %s file dir\n", argv[0]);
    return -1;
  }

  string dirname = argv[2];

  strncpy(imagename, argv[1], 512);
  cv::Mat img = cv::imread(imagename, 1);
  if(img.empty()) {
    fprintf(stderr, "%s is empty\n", imagename);
    return -1;
  }

  double scale = img.rows / 200.0; // ★横の解像度200pixelsにする
  cv::Mat gray, smallImg(cv::saturate_cast<int>(img.rows/scale), cv::saturate_cast<int>(img.cols/scale), CV_8UC1);

  // グレースケール画像に変換
  cv::cvtColor(img, gray, CV_BGR2GRAY);
  // 処理時間短縮のために画像を縮小
  cv::resize(gray, smallImg, smallImg.size(), 0, 0, cv::INTER_LINEAR);
  cv::equalizeHist( smallImg, smallImg);

  // 分類器の読み込み
  std::string cascadeName = "./haarcascade_frontalface_alt.xml"; // Haar-like
  cv::CascadeClassifier cascade;
  if(!cascade.load(cascadeName)) {
    std::cerr << "failed to load " << cascadeName << endl;
    return -1;
  }

  std::vector<cv::Rect> faces;
  /// マルチスケール（顔）探索xo
  // 画像，出力矩形，縮小スケール，最低矩形数，（フラグ），最小矩形
  cascade.detectMultiScale(smallImg, faces,
      1.1, 2,
      CV_HAAR_SCALE_IMAGE,
      cv::Size(30, 30));

  if (faces.size() != 1) {
    cerr << "faces are too many or too few faces: " << faces.size() << endl;
    return -1;
  }

  cv::Rect faceRect = faces.at(0);

  // 眼の検出
  std::string nested_cascadeName = "./haarcascade_eye.xml";
  cv::CascadeClassifier nested_cascade;
  if(!nested_cascade.load(nested_cascadeName)) {
    std::cerr << "failed to load " << nested_cascadeName << std::endl;
    return -1;
  }

  int r_count = 0;
  int nr_count = 0;
  cv:: Mat smallImgROI;
  std::vector<cv::Rect> nestedObjects;

  // 検出結果（顔）の描画
  cv::Point face_center;
  int face_radius;

  r_count++;

  smallImgROI = smallImg(faceRect);
  /// マルチスケール（目）探索
  // 画像，出力矩形，縮小スケール，最低矩形数，（フラグ），最小矩形
  nested_cascade.detectMultiScale(smallImgROI, nestedObjects,
      1.1, 3,
      CV_HAAR_SCALE_IMAGE, 
      cv::Size(10,10));

  // 検出結果（目）の描画
  std::vector<cv::Rect>::const_iterator nr = nestedObjects.begin();
  for(; nr != nestedObjects.end(); ++nr) {
    nr_count++;
  }

  cerr << "faces: " << r_count  << endl;
  cerr << "eyes: " << nr_count << endl;

  // 一人が写っている写真の場合、顔１目２
  if( r_count == 1 && nr_count == 2 ) {
    // 画像を保存する
    // 顔領域のみを保存 ROIで
    cv::imwrite(dirname + "/face.jpg", smallImgROI);
    cv::Rect eyeRect1 = nestedObjects.at(0);
    cv::Rect eyeRect2 = nestedObjects.at(1);
    cv::Rect eyeAbsRect1(faceRect.x + eyeRect1.x, faceRect.y + eyeRect1.y, eyeRect1.width, eyeRect1.height);
    cv::Rect eyeAbsRect2(faceRect.x + eyeRect2.x, faceRect.y + eyeRect2.y, eyeRect2.width, eyeRect2.height);
    cv::Mat left_eye, right_eye;

    if (eyeRect1.x < eyeRect2.x) {
      left_eye = smallImg(eyeAbsRect2);
      right_eye = smallImg(eyeAbsRect1);
    }
    else {
      left_eye = smallImg(eyeAbsRect1);
      right_eye = smallImg(eyeAbsRect2);
    }
    cv::imwrite(dirname + "/left_eye.jpg",left_eye);
    cv::imwrite(dirname + "/right_eye.jpg", right_eye);
  }
}
