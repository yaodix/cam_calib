#include <vector>

#include <opencv2/opencv.hpp>


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

	// 在原图（未去畸变）上提取棋盘格点
    std::string src_path = "./calib_data/left05.jpg";
		cv::Mat test_gray_img = cv::imread(src_path, cv::IMREAD_GRAYSCALE);
    std::vector<cv::Point2f> corner;									// 某一副图像找到的角点
		cv::undistort(inImage, rectify_img, intrinsic, distortion_coeff);

		cv::Mat rvec, tvec;
// objRealPoints是distance_corners对应得世界坐标系物体坐标,rvec,tvec是该objRealPoints平面对相机外参
// Output rotation vector (see @ref Rodrigues ) that, together with tvec, brings points from
// the model coordinate system to the camera coordinate system.
		solvePnP(objRealPoints, distance_corners, intrinsic, distortion_coeff, rvec,  tvec);
		cv::Mat R;
		cv::Rodrigues(rvec, R); // R is 3x3

		cv::Mat h(3, 3, CV_64FC1);
		cv::Mat pt_in_img(3,1,CV_64FC1);  // 注意这里是畸变矫正后图像rectify_img上的点
		pt_in_img.at<double>(0, 0) = 281;
		pt_in_img.at<double>(1, 0) = 273;
		pt_in_img.at<double>(2, 0) = 1.;
		h(cv::Range(0, 3), cv::Range(0, 2)) = R(cv::Range(0, 3), cv::Range(0, 2))*1;
		h(cv::Range(0, 3), cv::Range(2, 3)) = tvec * 1;
		cv::Mat w_coor = h.inv() * (intrinsic.inv() * pt_in_img);
		double x_world = w_coor.at<double>(0, 0) / w_coor.at<double>( 2,0);
		double y_world = w_coor.at<double>(1, 0) / w_coor.at<double>( 2,0);

		std::cout << 
}

	//R = R.t();  // rotation of inverse
	//tvec = -R * tvec; // translation of inverse,相机坐标原点在世界坐标系位置

int main() {

	return 0;
}