// 1. 投影功能
// 1.1 投影到畸变图像
// 1.2 投影到去畸变图像

#include <iostream>
#include <vector>
#include <string>

#include "opencv2/opencv.hpp"

const int boardWidth = 9;								  // 横向的角点数目
const int boardHeight = 6;								// 纵向的角点数据
const int squareSize = 25;								// 标定板黑白格子的大小，单位 mm

std::vector<cv::Mat> LoadParams(const std::string& file_path) {
	std::vector<cv::Mat> data;
	cv::FileStorage fs;
	fs.open(file_path, cv::FileStorage::READ);
	int cnt = 0;
	fs["mat_cnt"] >> cnt;
	for (int i =0; i < cnt; ++i) {
		cv::Mat temp;
		fs["data_" + std::to_string(i)] >> temp;
		data.push_back(temp);
	}
	return data;
}


void test_projectPoints() {
	// 世界坐标系投影到畸变图像
	std::string src_path = "./calib_data/left05.jpg";
	cv::Mat intrinsic = LoadParams("./workspace/intrinsic.yaml").front();
	cv::Mat distortion_coeff = LoadParams("./workspace/distortion_coeff.yaml").front();
	std::vector<cv::Mat> rvecs = LoadParams("./workspace/rvecs.yaml");
	std::vector<cv::Mat> tvecs = LoadParams("./workspace/tvecs.yaml");

	cv::Mat test_img = cv::imread(src_path);
	cv::Mat test_gray_img = cv::imread(src_path, cv::IMREAD_GRAYSCALE);

	cv::Mat undistort_img1;
	// 畸变矫正方法1-undistort
	cv::undistort(test_img, undistort_img1, intrinsic, distortion_coeff);

	std::vector<cv::Point3f> world_points;
	for (int rowIndex = 0; rowIndex < boardHeight; rowIndex++) { // boardheight=6;
		for (int colIndex = 0; colIndex < boardWidth; colIndex++) { // boardWidth = 9;		
			world_points.push_back(cv::Point3f(rowIndex * squareSize, colIndex * squareSize, 0));  
		}
	}

	int img_index = 4;
	std::vector<cv::Point2f> project_points;
	cv::projectPoints(world_points, rvecs[img_index], tvecs[img_index],
		intrinsic, distortion_coeff, project_points);

	for (auto& pt : project_points) {
		circle(test_img, pt, 3, cv::Scalar(255, 0, 0), 2);
	}
	cv::circle(test_img, project_points.front(), 3, cv::Scalar(0, 0, 250), 2);
	// cv::imshow("project points distort img", test_img);
	// cv::waitKey();
	// 世界坐标系点投影到去畸变图像
		std::vector<cv::Point2f> project_points2;
	cv::projectPoints(world_points, rvecs[img_index], tvecs[img_index],
		intrinsic, cv::Mat(), project_points2);
	
	for (auto& pt : project_points2) {
		circle(undistort_img1, pt, 3, cv::Scalar(255, 0, 0), 2);
	}
	cv::circle(undistort_img1, project_points2.front(), 3, cv::Scalar(0, 0, 250), 2);
	cv::imshow("project points undistort img", undistort_img1);
	cv::waitKey();
}

void test_projectCube() {
// AR投影立方体到棋盘格上
	std::string src_path = "./calib_data/left05.jpg";
	cv::Mat intrinsic = LoadParams("./workspace/intrinsic.yaml").front();
	cv::Mat distortion_coeff = LoadParams("./workspace/distortion_coeff.yaml").front();
	std::vector<cv::Mat> rvecs = LoadParams("./workspace/rvecs.yaml");
	std::vector<cv::Mat> tvecs = LoadParams("./workspace/tvecs.yaml");
	int img_index = 4;
	cv::Mat test_img = cv::imread(src_path);
	// opencv 4.5+
	// cv::drawFrameAxes(test_img, intrinsic, distortion_coeff, rvecs[img_index], tvecs[img_index], 2 * squareSize);
	
	int len_of_box = 1. * squareSize;
	cv::Point3d origin_point(0, 0, 0), point1(len_of_box, 0,0), point3(len_of_box, len_of_box ,0), point2(0, len_of_box, 0);
	cv::Point3d origin_point2(0, 0, len_of_box), point4(len_of_box, 0, len_of_box),
		point6(len_of_box, len_of_box, len_of_box), point5(0, len_of_box, len_of_box);

	std::vector<cv::Point3d> bbox{ origin_point, point1, point3, point2, origin_point2 , point4, point6 ,point5 };
	std::vector<cv::Point2d> bbox_plane;
	cv::projectPoints(bbox,rvecs[img_index],tvecs[img_index], intrinsic, distortion_coeff, bbox_plane);

	for (int i = 0; i < 4; i++) {
		cv::line(test_img, bbox_plane[i], bbox_plane[(i + 1) % 4], cv::Scalar(0, 0, 255), 3);
		cv::line(test_img, bbox_plane[i + 4], bbox_plane[(i + 4 + 1) == 8 ? 4 : i + 5], cv::Scalar(0, 255, 0), 3);
		cv::line(test_img, bbox_plane[i], bbox_plane[i + 4], cv::Scalar(255, 0, 0), 3);
	}
	cv::imshow("show box", test_img);
	cv::waitKey();
}

void test_CuboxAddARGame() {

}

int main() {
	test_projectPoints();
	// test_projectCube();
	return 0;
}



