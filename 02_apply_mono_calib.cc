#include <iostream>
#include <vector>
#include <string>


#include "opencv2/opencv.hpp"

// 1.图像去畸变功能
// 理解 remap，畸变图像与非畸变图像上点相互转换
 
// 2. 局部图像去畸变

// 3. 投影功能
// 3.1 投影到畸变图像
// 3.2 投影到去畸变图像


// 4. 旋转相机标定



int test_undistort() {
	std::string src_path = "./data/left05.jpg";
	cv::Mat src_img = cv::imread(src_path);
	cv::Mat src_gray_img = cv::imread(src_path, cv::IMREAD_GRAYSCALE);

	cv::Mat intrinsic = 


}

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

