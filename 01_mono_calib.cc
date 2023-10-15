// 单目相机标定示例
// 只支持opencv3.0及之后的版本

#include <iostream>
#include <vector>
#include <string>

#include "opencv2/opencv.hpp"
using namespace std;
using namespace cv;

const int imageWidth = 640;								// 摄像头的分辨率
const int imageHeight = 480;
const int boardWidth = 9;								  // 横向的角点数目
const int boardHeight = 6;								// 纵向的角点数据
const int boardCornerNum = boardWidth * boardHeight;		// 总的角点数据
const int frameNumber =13;								// 相机标定时需要采用的图像帧数
const int squareSize = 25;								// 标定板黑白格子的大小，单位 mm
const Size boardSize = Size(boardWidth, boardHeight);

Mat intrinsic;											  // 相机内参数
Mat distortion_coeff;							    // 相机畸变参数
vector<Mat> rvecs;									  // 旋转向量
vector<Mat> tvecs;									  // 平移向量
vector<vector<Point2f>> corners;			// 各个图像找到的角点的集合 和objRealPoint 一一对应
vector<vector<Point3f>> objRealPoint;	// 各副图像的角点的实际物理坐标集合

vector<Point2f> corner;									// 某一副图像找到的角点
Mat rgbImage, grayImage;

/*计算标定板上棋盘格的实际物理坐标--该坐标是世界坐标系*/
void calRealPoint(vector<vector<Point3f>>& obj, int boardwidth, int boardheight, int imgNumber, int squaresize) {
	vector<Point3f> imgpoint;
	for (int rowIndex = 0; rowIndex < boardheight; rowIndex++) { // boardheight=6;
		for (int colIndex = 0; colIndex < boardwidth; colIndex++) { // boardWidth = 9;		
			imgpoint.push_back(Point3f(rowIndex * squaresize, colIndex * squaresize, 0));  
		}
	}
	for (int imgIndex = 0; imgIndex < imgNumber; imgIndex++) {
		obj.push_back(imgpoint);
	}
}

/*设置相机的初始参数 也可以不估计*/
void guessCameraParam() {
	/*分配内存*/
	intrinsic.create(3, 3, CV_64FC1);
	distortion_coeff.create(5, 1, CV_64FC1);

	/*
	fx 0 cx
	0 fy cy
	0 0  1
	*/
	intrinsic.at<double>(0, 0) = 640;    // fx		
	intrinsic.at<double>(0, 2) = 320;   // cx
	intrinsic.at<double>(1, 1) = 640;   // fy
	intrinsic.at<double>(1, 2) = 240;   // cy

	intrinsic.at<double>(0, 1) = 0;
	intrinsic.at<double>(1, 0) = 0;
	intrinsic.at<double>(2, 0) = 0;
	intrinsic.at<double>(2, 1) = 0;
	intrinsic.at<double>(2, 2) = 1;

	/*
	k1 k2 p1 p2 p3
	*/
	distortion_coeff.at<double>(0, 0) = 0;   // k1
	distortion_coeff.at<double>(1, 0) = 0.;  // k2
	distortion_coeff.at<double>(2, 0) = 0.;  // p1
	distortion_coeff.at<double>(3, 0) = 0.;  // p2
	distortion_coeff.at<double>(4, 0) = 0.;	 // p3
}

int SaveYaml(const std::string& file_path, const std::vector<cv::Mat>& data) {
  string filename = file_path;
  FileStorage fs(filename, FileStorage::WRITE);
  for (int i = 0; i < data.size(); ++i) {
    fs << "data_"+std::to_string(i) << data[i];
  }

  fs.release();
  std::cout << file_path << " write done" << std::endl;
}

void outputCameraParam(bool save_params = true) {  
  if (save_params) {
    SaveYaml("./workspace/intrinsic.yaml", std::vector<cv::Mat>{intrinsic});
    SaveYaml("./workspace/distortion_coeff.yaml", std::vector<cv::Mat>{distortion_coeff});
    SaveYaml("./workspace/rvecs.yaml", rvecs);
    SaveYaml("./workspace/tvecs.yaml", tvecs);

  }
	/*输出数据*/
	cout << "fx :" << intrinsic.at<double>(0, 0) << endl << "fy :" << intrinsic.at<double>(1, 1) << endl;
	cout << "cx :" << intrinsic.at<double>(0, 2) << endl << "cy :" << intrinsic.at<double>(1, 2) << endl;

	cout << "k1 :" << distortion_coeff.at<double>(0, 0) << endl;
	cout << "k2 :" << distortion_coeff.at<double>(1, 0) << endl;
	cout << "p1 :" << distortion_coeff.at<double>(2, 0) << endl;
	cout << "p2 :" << distortion_coeff.at<double>(3, 0) << endl;
	cout << "k3 :" << distortion_coeff.at<double>(4, 0) << endl;
}

// 计算重投影误差
// objectPoints: 标定板世界坐标系点
// imagePoints: 标定板图片查找到的棋盘格交点
double computeReprojectionErrors(
    const vector<vector<Point3f> >& objectPoints,
    const vector<vector<Point2f> >& imagePoints,
    const vector<Mat>& rvecs, const vector<Mat>& tvecs,
    const Mat& cameraMatrix, const Mat& distCoeffs,
    vector<float>& perViewErrors) {
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
	return std::sqrt(totalErr / totalPoints);
}


int main(int argc, char *arg[]) {
	Mat img;
	int goodFrameCount = 0;
	int num = 0;
	namedWindow("chessboard");
	cout << "按Q退出 ..." << endl;
	char filename[100],filename2[100];
	while (goodFrameCount < frameNumber) {
		sprintf(filename, "./calib_data/left%02d.jpg", goodFrameCount + 1);
		// sprintf_s(filename, "./calib_data/right%02d.jpg", goodFrameCount + 1);
			cout << filename << endl;
		rgbImage = imread(filename);
		cvtColor(rgbImage, grayImage, COLOR_BGR2GRAY);
		bool isFind = findChessboardCorners(grayImage, boardSize, corner);
		if (isFind == true) {	//所有角点都被找到 说明这幅图像是可行的
			/*
			Size(5,5) 搜索窗口的一半大小
			Size(-1,-1) 死区的一半尺寸
			TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 20, 0.1)迭代终止条件
			*/
			cornerSubPix(grayImage, corner, Size(5, 5), Size(-1, -1), 
					TermCriteria(TermCriteria::Type::COUNT | TermCriteria::Type::COUNT, 20, 0.1));
      cv::Mat draw_chessbaord_img;
      draw_chessbaord_img = rgbImage.clone();
			drawChessboardCorners(draw_chessbaord_img, boardSize, corner, isFind);
      cv::Mat find_corner_img;
      cv::hconcat(std::vector<cv::Mat>{rgbImage, draw_chessbaord_img}, find_corner_img);
			imshow("chessboard", find_corner_img);
			corners.push_back(corner);
			goodFrameCount++;
			cout << "The image is good" << endl;
		}	else	{
			cout << "The image is bad please try again" << endl;
		}
		//	cout << "Press any key to continue..." << endl;
			waitKey(100);  // 显示延时

		if (waitKey(10) == 'q')	{
			break;
		}
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
	double rms = calibrateCamera(objRealPoint, corners, Size(imageWidth, imageHeight), intrinsic, distortion_coeff, rvecs, tvecs);
  std::cout << "rms " << rms << std::endl;
	cout << "calibration successful" << endl;
	/*保存并输出参数*/
	outputCameraParam();
	cout << "param out successful" << endl;

	/*显示畸变校正效果*/
  Mat inImage_color = imread("./calib_data/left08.jpg");
  cv::Mat rectify_img;
	cv::undistort(inImage_color, rectify_img, intrinsic, distortion_coeff);

  std::vector<float> reprojErrs;
	double totalAvgErr = computeReprojectionErrors(objRealPoint, corners,
		rvecs, tvecs, intrinsic, distortion_coeff, reprojErrs);

  auto min_max_pair = std::minmax_element(reprojErrs.begin(), reprojErrs.end());
	cout << "重投影误差：" << totalAvgErr << ", max error " << *min_max_pair.first << ", min error " << (*min_max_pair.second) << endl;
  cv::Mat hor_img;
  cv::hconcat(std::vector<cv::Mat>{inImage_color, rectify_img}, hor_img);
	imshow("rectify_img", hor_img);
	waitKey(0);  // 程序暂停
	return 0;
}
