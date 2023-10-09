#include <iostream>
#include <vector>

#include "opencv2/opencv.hpp"

// 标定单目相机，size:640*480
int CalibMono() {

}

int main() {

  return 0;
}

// monocalib.cpp : 单目相机标定过程示例
//

//只支持opencv3.0及之后的版本
//#include "stdafx.h"

//#include <opencv.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

const int imageWidth = 640;								//摄像头的分辨率
const int imageHeight = 480;
const int boardWidth = 9;								//横向的角点数目
const int boardHeight = 6;								//纵向的角点数据
const int boardCornerNum = boardWidth * boardHeight;		//总的角点数据
const int frameNumber =13;								//相机标定时需要采用的图像帧数
const int squareSize = 25;								//标定板黑白格子的大小 单位mm
const Size boardSize = Size(boardWidth, boardHeight);	//
	
Mat intrinsic;											//相机内参数
Mat distortion_coeff;									//相机畸变参数
vector<Mat> rvecs;									    //旋转向量
vector<Mat> tvecs;										//平移向量
vector<vector<Point2f>> corners;						//各个图像找到的角点的集合 和objRealPoint 一一对应
vector<vector<Point3f>> objRealPoint;					//各副图像的角点的实际物理坐标集合


vector<Point2f> corner;									//某一副图像找到的角点

Mat rgbImage, grayImage;

/*计算标定板上模块的实际物理坐标*/
void calRealPoint(vector<vector<Point3f>>& obj, int boardwidth,int boardheight, int imgNumber, int squaresize)
{
//	Mat imgpoint(boardheight, boardwidth, CV_32FC3,Scalar(0,0,0));
	vector<Point3f> imgpoint;
	for (int rowIndex = 0; rowIndex < boardheight; rowIndex++)	//boardheight=6;
	{
		for (int colIndex = 0; colIndex < boardwidth; colIndex++) //boardWidth = 9;		
		{
		//	imgpoint.at<Vec3f>(rowIndex, colIndex) = Vec3f(rowIndex * squaresize, colIndex*squaresize, 0);
			imgpoint.push_back(Point3f(rowIndex * squaresize, colIndex * squaresize, 0));
		}
	}
	for (int imgIndex = 0; imgIndex < imgNumber; imgIndex++)
	{
		obj.push_back(imgpoint);
	}
}

/*设置相机的初始参数 也可以不估计*/
void guessCameraParam(void )
{
	/*分配内存*/
	intrinsic.create(3, 3, CV_64FC1);
	distortion_coeff.create(5, 1, CV_64FC1);

	/*
	fx 0 cx
	0 fy cy
	0 0  1
	*/
	intrinsic.at<double>(0,0) = 600;   //fx		
	intrinsic.at<double>(0, 2) = 320;   //cx
	intrinsic.at<double>(1, 1) = 600;   //fy
	intrinsic.at<double>(1, 2) = 240;   //cy

	intrinsic.at<double>(0, 1) = 0;
	intrinsic.at<double>(1, 0) = 0;
	intrinsic.at<double>(2, 0) = 0;
	intrinsic.at<double>(2, 1) = 0;
	intrinsic.at<double>(2, 2) = 1;

	/*
	k1 k2 p1 p2 p3
	*/
	distortion_coeff.at<double>(0, 0) = -0.193740;  //k1
	distortion_coeff.at<double>(1, 0) = -0.378588;  //k2
	distortion_coeff.at<double>(2, 0) = 0.028980;   //p1
	distortion_coeff.at<double>(3, 0) = 0.008136;   //p2
	distortion_coeff.at<double>(4, 0) = 0;		  //p3
}

void outputCameraParam(void )
{
	/*保存数据*/
	//cvSave("cameraMatrix.xml", &intrinsic);
	//cvSave("cameraDistoration.xml", &distortion_coeff);
	//cvSave("rotatoVector.xml", &rvecs);
	//cvSave("translationVector.xml", &tvecs);
	/*输出数据*/
	cout << "fx :" << intrinsic.at<double>(0, 0) << endl << "fy :" << intrinsic.at<double>(1, 1) << endl;
	cout << "cx :" << intrinsic.at<double>(0, 2) << endl << "cy :" << intrinsic.at<double>(1, 2) << endl;

	cout << "k1 :" << distortion_coeff.at<double>(0, 0) << endl;
	cout << "k2 :" << distortion_coeff.at<double>(1, 0) << endl;
	cout << "p1 :" << distortion_coeff.at<double>(2, 0) << endl;
	cout << "p2 :" << distortion_coeff.at<double>(3, 0) << endl;
	cout << "k3 :" << distortion_coeff.at<double>(4, 0) << endl;
}

//计算重投影误差
double computeReprojectionErrors(
	const vector<vector<Point3f> >& objectPoints,
	const vector<vector<Point2f> >& imagePoints,
	const vector<Mat>& rvecs, const vector<Mat>& tvecs,
	const Mat& cameraMatrix, const Mat& distCoeffs,
	vector<float>& perViewErrors)
{
	vector<Point2f> imagePoints2;
	imagePoints2.reserve(imagePoints[0].size());
	int i, totalPoints = 0;
	double totalErr = 0, err;
	perViewErrors.resize(objectPoints.size());

	for (i = 0; i < (int)objectPoints.size(); i++)
	{
		projectPoints(Mat(objectPoints[i]), rvecs[i], tvecs[i],
			cameraMatrix, distCoeffs, imagePoints2);
		err = norm(Mat(imagePoints[i]), Mat(imagePoints2), cv::NORM_L2);
		int n = (int)objectPoints[i].size();
		perViewErrors[i] = (float)std::sqrt(err * err / n);
		totalErr += err * err;
		totalPoints += n;
	}
	//cout<<double( sqrt(totalErr / totalPoints))<<endl;
	return std::sqrt(totalErr / totalPoints);
}

void myundistort(const Mat& src, Mat& dst, const Mat& _cameraMatrix,
	const Mat& _distCoeffs, const Mat& _newCameraMatrix) {
	dst.create(src.size(), src.type());
	CV_Assert(dst.data != src.data);

	int stripe_size0 = std::min(std::max(1, (1 << 12) / std::max(src.cols, 1)), src.rows);
	Mat map1(stripe_size0, src.cols, CV_16SC2), map2(stripe_size0, src.cols, CV_16UC1);

	Mat_<double> A, distCoeffs, Ar, I = Mat_<double>::eye(3, 3);

	_cameraMatrix.convertTo(A, CV_64F);
	if (_distCoeffs.data)
		distCoeffs = Mat_<double>(_distCoeffs);
	else {
		distCoeffs.create(5, 1);
		distCoeffs = 0.;
	}

	if (_newCameraMatrix.data)
		_newCameraMatrix.convertTo(Ar, CV_64F);
	else
		A.copyTo(Ar);

	double v0 = Ar(1, 2);
	for (int y = 0; y < src.rows; y += stripe_size0) {
		int stripe_size = std::min(stripe_size0, src.rows - y);
		Ar(1, 2) = v0 - y;
		Mat map1_part = map1.rowRange(0, stripe_size),
			map2_part = map2.rowRange(0, stripe_size),
			dst_part = dst.rowRange(y, y + stripe_size);

		initUndistortRectifyMap(A, distCoeffs, I, Ar, Size(src.cols, stripe_size),
			map1_part.type(), map1_part, map2_part);
		remap(src, dst_part, map1_part, map2_part, INTER_LINEAR, BORDER_CONSTANT);
	}
}


int  main(int argc, char *arg[] )
{
	Mat img;
	int goodFrameCount = 0;
	int num = 0;
	namedWindow("chessboard");
	cout << "按Q退出 ..." << endl;
		char filename[100],filename2[100];
	while (goodFrameCount < frameNumber)
	{
		//sprintf_s(filename, "C:\\C++Projects\\pic\\mono_imgL\\left%02d.jpg", goodFrameCount + 1);
		sprintf_s(filename, "C:\\C++Projects\\pic\\mono_imgR\\right%02d.jpg", goodFrameCount + 1);
		//	cout << filename << endl;
		rgbImage = imread(filename);
		cvtColor(rgbImage, grayImage, COLOR_BGR2GRAY);
		imshow("Camera", grayImage);
		bool isFind = findChessboardCorners(grayImage, boardSize, corner);
		if (isFind == true)	//所有角点都被找到 说明这幅图像是可行的
		{
			/*
			Size(5,5) 搜索窗口的一半大小
			Size(-1,-1) 死区的一半尺寸
			TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 20, 0.1)迭代终止条件
			*/
			cornerSubPix(grayImage, corner, Size(5, 5), Size(-1, -1), 
					TermCriteria(TermCriteria::Type::COUNT | TermCriteria::Type::COUNT, 20, 0.1));
			drawChessboardCorners(rgbImage, boardSize, corner, isFind);
			imshow("chessboard", rgbImage);
			corners.push_back(corner);
			goodFrameCount++;
			cout << "The image is good" << endl;
		}
		else
		{
			cout << "The image is bad please try again" << endl;
		}
		//	cout << "Press any key to continue..." << endl;
		//	waitKey(0);

		if (waitKey(10) == 'q')
		{
			break;
		}
		//	imshow("chessboard", rgbImage);
	}

	/*
	图像采集完毕 接下来开始摄像头的校正
	calibrateCamera()
	输入参数 objectPoints  角点的实际物理坐标
			 imagePoints   角点的图像坐标
			 imageSize	   图像的大小
	输出参数
			 cameraMatrix  相机的内参矩阵
			 distCoeffs	   相机的畸变参数
			 rvecs		   旋转矢量(外参数)
			 tvecs		   平移矢量(外参数）
	*/

	/*设置实际初始参数 根据calibrateCamera来 如果flag = 0 也可以不进行设置*/
	guessCameraParam();
	cout << "guess successful" << endl;
	/*计算实际的校正点的三维坐标*/
	calRealPoint(objRealPoint, boardWidth, boardHeight, frameNumber, squareSize);
	cout << "cal real successful" << endl;
	/*标定摄像头*/
	//calibrateCamera(objRealPoint, corners, Size(imageWidth, imageHeight), intrinsic, distortion_coeff, rvecs, tvecs, CV_CALIB_FIX_PRINCIPAL_POINT);
	double rms = calibrateCamera(objRealPoint, corners, Size(imageWidth, imageHeight), intrinsic, distortion_coeff, rvecs, tvecs);

	cout << "calibration successful" << endl;
	/*保存并输出参数*/
	outputCameraParam();
	cout << "out successful" << endl;


	/*显示畸变校正效果*/
	Mat inImage_color = imread("C:\\C++Projects\\pic\\mono_imgR\\right01.jpg");
	Mat inImage = imread("C:\\C++Projects\\pic\\mono_imgR\\right01.jpg", cv::IMREAD_GRAYSCALE);
	Mat cImage, cImage2, map1,map2;
	int i = 0;
	vector<Point2f>  image_points2;
	vector<Point3f> tempPointSet = objRealPoint[i];

	float scale = 1.5;  // 可任意设定
	cv::Size new_size, new_size2;
	new_size2.width = inImage.size().width * scale;
	new_size2.height = inImage.size().height * scale;

	cv::Mat distort_zero = distortion_coeff.clone();
	distort_zero.setTo(0);
	// alpha 设置为1，all the source image pixels are retained in the undistorted image
	cv::Mat new_cam_matrix2 = cv::getOptimalNewCameraMatrix(intrinsic, distortion_coeff, inImage.size(), 1, new_size2);
	cv::initUndistortRectifyMap(intrinsic, distort_zero, cv::Mat(), new_cam_matrix2, new_size2, CV_32FC1, map1, map2);
	cv::remap(inImage, cImage, map1, map2, cv::InterpolationFlags::INTER_LINEAR, cv::BorderTypes::BORDER_CONSTANT, 0);

	//undistort(inImage, cImage2, intrinsic, distortion_coeff);

	vector<Point2f>  imagePointsPro, imagePointsPro2;
	cv::projectPoints(Mat(objRealPoint[i]), rvecs[i], tvecs[i], new_cam_matrix2, distortion_coeff, imagePointsPro2);

	for (auto pt : imagePointsPro2) {
		circle(cImage, pt, 3, cv::Scalar(255, 0, 0), 2);
	}

	vector<float> reprojErrs;
	reprojErrs.resize(objRealPoint.size());
	double totalAvgErr = 0;

	vector<cv::Point2f> distance_corners;
	bool isFind = findChessboardCorners(inImage, boardSize, distance_corners);
	cv::Mat rectify_img;
	cv::undistort(inImage, rectify_img, intrinsic, distortion_coeff);
	// 
// pnp问题
	Mat rvec, tvec;
	int index = 2;
	solvePnP(objRealPoint[index], distance_corners, intrinsic, distortion_coeff, rvec, tvec);
	cv::Mat R;
	cv::Rodrigues(rvec, R); // R is 3x3

	//R = R.t();  // rotation of inverse
	//tvec = -R * tvec; // translation of inverse,相机坐标原点在世界坐标系位置

	/*  点从世界坐标系到相机坐标系旋转矩阵
	cv::Mat R;
	cv::Rodrigues(rvec, R); // R is 3x3
	cv::Mat T(4, 4, R.type()); // T is 4x4
	T(cv::Range(0, 3), cv::Range(0, 3)) = R * 1; // copies R into T
	T(cv::Range(0, 3), cv::Range(3, 4)) = tvec * 1; // copies tvec into T
	// fill the last row of T (NOTE: depending on your types, use float or double)
	double* p = T.ptr<double>(3);
	p[0] = p[1] = p[2] = 0; p[3] = 1;


	cv::Mat pos_w(4, 1, CV_64FC1);
	pos_w.at<double>(0, 0) = 25;
	pos_w.at<double>(1, 0) = 25;
	pos_w.at<double>(2, 0) = 0;
	pos_w.at<double>(3, 0) = 1;

	cv::Mat pos_cam = T * pos_w;

	cv::Mat pos_c(3, 1, CV_64FC1);
	pos_c.at<double>(0, 0) = pos_cam.at<double>(0, 0);
	pos_c.at<double>(1, 0) = pos_cam.at<double>(1, 0);
	pos_c.at<double>(2, 0) = pos_cam.at<double>(2, 0);

	cv::Mat pos_img = intrinsic * pos_c;

	double u = pos_img.at<double>(0, 0) / pos_img.at<double>(2, 0);
	double v = pos_img.at<double>(1, 0) / pos_img.at<double>(2, 0);
	//*/

	cv::Mat h(3, 3, CV_64FC1);
	cv::Mat pt_in_img(3,1,CV_64FC1);
	pt_in_img.at<double>(0, 0) = 281;
	pt_in_img.at<double>(1, 0) = 273.138898;
	pt_in_img.at<double>(2, 0) = 1.;

	h(cv::Range(0, 3), cv::Range(0, 2)) = R(cv::Range(0, 3), cv::Range(0, 2))*1;
	h(cv::Range(0, 3), cv::Range(2, 3)) = tvec * 1;


	cv::Mat w_coor = h.inv() * (intrinsic.inv() * pt_in_img);

	double x = w_coor.at<double>(0, 0) / w_coor.at<double>( 2,0);
	double y = w_coor.at<double>(1, 0) / w_coor.at<double>( 2,0);

	cv::Point p(281, 273);
	cv::Vec3d pvec((p.x - intrinsic.at<double>(0, 2)) / intrinsic.at<double>(0, 0),
		(p.y - intrinsic.at<double>(1, 2)) / intrinsic.at<double>(1, 1), 1.0);

	cv::Mat co_pvec = R * pvec;
	//co_pvec.normalize();
	//cv::normalize(co_pvec, co_pvec);

	double z = 0;
	double x1 = tvec.at<double>(0, 0)
		+ co_pvec.at<double>(0,0) * (z - tvec.at<double>(2, 0)) / co_pvec.at<double>(2, 0);
	double y1 = tvec.at<double>(1, 0)
		+ co_pvec.at<double>(1, 0) * (z - tvec.at<double>(2, 0)) / co_pvec.at<double>(2, 0);


	drawFrameAxes(inImage_color, intrinsic, distortion_coeff, rvec, tvec, 2 * squareSize);
	int len_of_box = squareSize;
	cv::Point3d origin_point(0, 0, 0), point1(len_of_box, 0,0), point3(len_of_box, len_of_box ,0), point2(0, len_of_box, 0);
	cv::Point3d origin_point2(0, 0, len_of_box), point4(len_of_box, 0, len_of_box),
		point6(len_of_box, len_of_box, len_of_box), point5(0, len_of_box, len_of_box);

	vector<cv::Point3d> bbox{ origin_point, point1, point3, point2, origin_point2 , point4, point6 ,point5 };
	vector<cv::Point2d> bbox_plane;
	projectPoints(bbox,rvec,tvec, intrinsic, distortion_coeff, bbox_plane);

	for (int i = 0; i < 4; i++) {
		cv::line(inImage_color, bbox_plane[i], bbox_plane[(i + 1) % 4], cv::Scalar(0, 0, 255), 1);
		cv::line(inImage_color, bbox_plane[i + 4], bbox_plane[(i + 4 + 1) == 8 ? 4 : i + 5], cv::Scalar(0, 255, 0), 1);
		cv::line(inImage_color, bbox_plane[i], bbox_plane[i + 4], cv::Scalar(255, 0, 0), 1);
	}

	totalAvgErr = computeReprojectionErrors(objRealPoint, corners,
		rvecs, tvecs, intrinsic, distortion_coeff, reprojErrs);

	cout << "重投影误差：" << totalAvgErr << endl;

	imshow("pointImage", cImage);
	waitKey(0);
	//system("pause");
	return 0;
}




