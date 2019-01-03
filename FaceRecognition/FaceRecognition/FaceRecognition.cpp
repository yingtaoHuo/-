// FaceRecognition.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include "pch.h"
#include <iostream>
#include <sstream>
#include <set>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>

using namespace std;
using namespace cv;
using namespace cv::face;

int studentNum;
set<string> nameResult;
vector<string> namelist;

void Distinguish()
{
	 //定义保存图片和标签的向量容器
	std::vector<Mat> images;
	std::vector<int> labels;
	//预测样本
	Mat src = imread("tmp.jpg", IMREAD_GRAYSCALE);
	//图像大小归一化
	cv::resize(src, src, Size(128, 128));

	Ptr<EigenFaceRecognizer> faceClass = EigenFaceRecognizer::create();
	//Ptr<FisherFaceRecognizer> fisherClass = FisherFaceRecognizer::create();
	//Ptr<LBPHFaceRecognizer> lpbhClass = LBPHFaceRecognizer::create();
	//加载分类器
	faceClass= EigenFaceRecognizer::load<EigenFaceRecognizer>("faceClass.xml");
	//fisherClass= FisherFaceRecognizer::load<FisherFaceRecognizer>("fisherClass.xml");
	//lpbhClass= LBPHFaceRecognizer::load<LBPHFaceRecognizer>("lpbhClass.xml");
	//使用训练好的分类器进行预测。
	int faceResult = faceClass->predict(src);
	if (faceResult >= 1 && faceResult <= studentNum) nameResult.insert(namelist[faceResult]);
	return;
}

void FindFaces(Mat src)
{
	Mat frame = src.clone();
	Mat facesROI;
	//图像缩放，采用双线性插值。
	//cv::resize(src,src,Size(128,128),0,0,cv::INTER_LINEAR);
	//图像灰度化。
	cv::cvtColor(src, src, COLOR_BGR2GRAY);
	//直方图均衡化，图像增强，暗的变亮，亮的变暗。
	cv::equalizeHist(src, src);
	//
	String face_cascade_name = "haarcascade_frontalface_alt.xml";
	String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
	CascadeClassifier face_cascade, eyes_cascade;
	if (!face_cascade.load(face_cascade_name))
	{
		//加载脸部分类器失败！
		return;
	}
	if (!eyes_cascade.load(eyes_cascade_name))
	{
		//加载眼睛分类器失败！
		return;
	}
	//存储找到的脸部矩形。
	std::vector<Rect> faces;
	face_cascade.detectMultiScale(src, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
	for (size_t i = 0; i < faces.size(); ++i)
	{
		//绘制矩形 BGR。
		rectangle(frame, faces[i], Scalar(0, 0, 255), 1);
		//截取人脸。
		facesROI = frame(faces[i]);
		//图像缩放。
		cv::resize(facesROI, facesROI, Size(128, 128), 0, 0, cv::INTER_LINEAR);
		//保存图像。
		cv::imwrite("tmp.jpg", facesROI);
		//识别
		Distinguish();
	}
	return;
}

void init()
{
	stringstream ss;
	String face_cascade_name = "haarcascade_frontalface_alt.xml";
	String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
	CascadeClassifier face_cascade, eyes_cascade;
	if (!face_cascade.load(face_cascade_name))
	{
		//加载脸部分类器失败！
		return;
	}
	if (!eyes_cascade.load(eyes_cascade_name))
	{
		//加载眼睛分类器失败！
		return;
	}
	for (int i = 1; i <= studentNum; i++) {
		for (int j = 1; j <= 4; j++) {
			ss.str("");
			ss << i << "/" << j << ".jpg";
			string tmp = "trainData/" + ss.str();
			Mat src = imread(tmp);

			Mat frame = src.clone();
			Mat facesROI;
			//图像缩放，采用双线性插值。
			//cv::resize(src,src,Size(128,128),0,0,cv::INTER_LINEAR);
			//图像灰度化。
			cv::cvtColor(src, src, COLOR_BGR2GRAY);
			//直方图均衡化，图像增强，暗的变亮，亮的变暗。
			cv::equalizeHist(src, src);
			//存储找到的脸部矩形。
			std::vector<Rect> faces;
			face_cascade.detectMultiScale(src, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
			for (size_t k = 0; k < faces.size(); ++k)
			{
				//绘制矩形 BGR。
				rectangle(frame, faces[k], Scalar(0, 0, 255), 1);
				facesROI = frame(faces[k]);
				//图像缩放。
				cv::resize(facesROI, facesROI, Size(128, 128), 0, 0, cv::INTER_LINEAR);
				//保存图像。
				cv::imwrite(tmp, facesROI);
			}
		}
	}
	return;
}

void train()
{
	std::vector<Mat> images;
	std::vector<int> labels;
	stringstream ss;
	//样本初始化
	init();
	//读取样本
	for (int i = 1; i <= studentNum; i++) {
		for (int j = 1; j <= 4; j++) {
			ss.str("");
			ss << i << "/" << j << ".jpg";
			string tmp = "trainData/"+ss.str();
			Mat src = imread(tmp,IMREAD_GRAYSCALE);
			Mat dst;
			//cv::imwrite("2.jpg", src);
			//图像大小归一化
			cv::resize(src, dst, Size(128, 128));
			//加入图像
			images.push_back(dst);
			//加入标签
			labels.push_back(i);
		}
	}
	Ptr<EigenFaceRecognizer> faceClass = EigenFaceRecognizer::create();
	//Ptr<FisherFaceRecognizer> fisherClass = FisherFaceRecognizer::create();
	//Ptr<LBPHFaceRecognizer> lpbhClass = LBPHFaceRecognizer::create();
	//训练
	faceClass->train(images, labels);
	//fisherClass->train(images, labels);
	//lpbhClass->train(images, labels);
	//保存训练的分类器
	faceClass->save("faceClass.xml");
	//fisherClass->save("fisherClass.xml");
	//lpbhClass->save("lpbhClass.xml");
}

int main(int argc,char* argv[])
{
	//读取学生名册
	ifstream namelistin("namelist.txt");
	//ofstream test("test.txt");
	//test << "123";
	namelist.push_back("");
	namelistin >> studentNum;
	string nametmp;
	for (int i = 1; i <= studentNum; i++) {
		namelistin >> nametmp;
		namelist.push_back(nametmp);
	}
	//学生名册读取完成，读取需要处理的照片进行处理
	int num = argc > 1 ? atoi(argv[1]) : 1;
	//输入参数为-1时进行训练
	if (num == -1) {
		train();
		return 0;
	}
	//否则进行识别
	stringstream ss;
	string srcName;
	for (int i = 1; i <= num; i++) {
		ss.str("");
		ss << i << ".jpg";
		srcName = ss.str();
		Mat src = imread(srcName);
		FindFaces(src);
	}
	//输出结果
	ofstream nameResultout("result.txt");
	for (auto i = nameResult.begin(); i != nameResult.end(); i++) {
		nameResultout << *i << endl;
	}
}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门提示: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
