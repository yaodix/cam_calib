#include <vector>

#include "opencv2/opencv.hpp"

// pnp问题
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

void test_caldistance() {
  std::vector<cv::Point2f> distance_corners;
	cv::Mat intrinsic = LoadParams("./workspace/intrinsic.yaml").front();
	cv::Mat distortion_coeff = LoadParams("./workspace/distortion_coeff.yaml").front();
	std::vector<cv::Mat> rvecs = LoadParams("./workspace/rvecs.yaml");
	std::vector<cv::Mat> tvecs = LoadParams("./workspace/tvecs.yaml");

	// 在原图（未去畸变）上提取棋盘格点
	std::string src_path = "./calib_data/left05.jpg";
	cv::Mat test_gray_img = cv::imread(src_path, cv::IMREAD_GRAYSCALE);
	cv::Mat test_img = cv::imread(src_path);
	cv::Mat rectify_img;
	cv::undistort(test_img, rectify_img, intrinsic, distortion_coeff);

	int img_index = 4;
	cv::Mat rvec = rvecs[img_index];
	cv::Mat tvec = tvecs[img_index];
	cv::Mat rot;
	cv::Rodrigues(rvec, rot);	
	
	// rot = rot.t();  // rotation of inverse
	// tvec = -rot * tvec; // translation of inverse,相机坐标原点在世界坐标系位置

	cv::Mat h(3, 3, CV_64FC1);  // 变换矩阵-外参数
	cv::Mat pt_in_img(3,1,CV_64FC1);  // 注意这里是畸变矫正后图像rectify_img上的点
	pt_in_img.at<double>(0, 0) = 410; // 取棋盘格点（1， 1,）
	pt_in_img.at<double>(1, 0) = 82;
	pt_in_img.at<double>(2, 0) = 1.;
	cv::circle(rectify_img, cv::Point(410, 82), 3, cv::Scalar(0, 0, 250), 2);

	h(cv::Range(0, 3), cv::Range(0, 2)) = rot(cv::Range(0, 3), cv::Range(0, 2))*1;
	h(cv::Range(0, 3), cv::Range(2, 3)) = tvec * 1;
	cv::Mat w_coor = h.inv() * (intrinsic.inv() * pt_in_img);
	double x_world = w_coor.at<double>(0, 0) / w_coor.at<double>( 2,0);  // 仅能获得归一化平面上点
	double y_world = w_coor.at<double>(1, 0) / w_coor.at<double>( 2,0);
	std::cout << "world dist " << x_world << " " << y_world << std::endl;

}


int main() {

	test_caldistance();
	return 0;
}