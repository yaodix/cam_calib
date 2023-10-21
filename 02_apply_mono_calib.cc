
// 1.图像去畸变功能
// 理解 remap，畸变图像与非畸变图像上点相互转换
 
// 2. 局部图像去畸变

#include <iostream>
#include <vector>
#include <string>

#include "opencv2/opencv.hpp"

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


int test_undistort() {
	std::string src_path = "./calib_data/left05.jpg";
	cv::Mat intrinsic = LoadParams("./workspace/intrinsic.yaml").front();
	cv::Mat distortion_coeff = LoadParams("./workspace/distortion_coeff.yaml").front();

	cv::Mat test_img = cv::imread(src_path);
	cv::Mat test_gray_img = cv::imread(src_path, cv::IMREAD_GRAYSCALE);

	cv::Mat undistort_img1, undistort_img2, u3;
	// 畸变矫正方法1-undistort
	cv::undistort(test_img, undistort_img1, intrinsic, distortion_coeff);

	// newCameraMatrix参数如何使用
	cv::Mat intrinsic_crop = intrinsic.clone();
	intrinsic_crop.at<double>(0, 2) = 200;
	cv::undistort(test_img, u3, intrinsic, distortion_coeff, intrinsic_crop);
	cv::Mat u4;
	cv::undistort(test_img, u4, intrinsic_crop, distortion_coeff);

	cv::Mat intrinsic_scale = intrinsic.clone();
	float scale = 0.5;
	intrinsic_scale.at<double>(0, 0) *= scale;
	intrinsic_scale.at<double>(0, 2) *= scale;
	intrinsic_scale.at<double>(1, 1) *= scale;
	intrinsic_scale.at<double>(1, 2) *= scale;
	cv::Mat u5;
	cv::undistort(test_img, u5, intrinsic, distortion_coeff, intrinsic_scale);

	// 畸变矫正方法2-remap
	cv::Mat map_x, map_y;
	cv::Mat intrinsic_copy = intrinsic.clone();
	cv::initUndistortRectifyMap(intrinsic, distortion_coeff, cv::Mat(), 
			intrinsic_copy, test_img.size(), CV_32FC1, map_x, map_y);
	cv::remap(test_img, undistort_img2, map_x, map_y, cv::InterpolationFlags::INTER_LINEAR,
			cv::BorderTypes::BORDER_CONSTANT, 0);

	// remap - resize
	cv::Mat u11, u12;
	cv::initUndistortRectifyMap(intrinsic, distortion_coeff, cv::Mat(), 
			intrinsic_scale, cv::Size(test_img.size().width * scale,test_img.size().height * scale) ,
			CV_32FC1, map_x, map_y);
	cv::remap(test_img, u11, map_x, map_y, cv::InterpolationFlags::INTER_LINEAR,
			cv::BorderTypes::BORDER_CONSTANT, 0);

	// remap - crop
	cv::initUndistortRectifyMap(intrinsic, distortion_coeff, cv::Mat(), 
			intrinsic_crop, cv::Size(test_img.size().width - (intrinsic.at<double>(0, 2) - 200) ,test_img.size().height) ,
			CV_32FC1, map_x, map_y);
	cv::remap(test_img, u12, map_x, map_y, cv::InterpolationFlags::INTER_LINEAR,
			cv::BorderTypes::BORDER_CONSTANT, 0);


	cv::Mat diff_im = undistort_img2 - undistort_img1;
	bool is_same = cv::sum(diff_im)[0] > 0 ? false : true;
	// 此处两种方法去畸变图像一般只有个别像素有微小差异，可以认为是相同的去畸变图像
	std::cout << "two undistort image same? " << is_same << std::endl;

	// getOptimalNewCameraMatrix 的使用
	// 对alpha参数的理解， 取值[0-1]保留有效像素的范围，for a better control over scaling. 
	cv::Mat u21, u22;
	cv::Mat new_cam_matrix_alpha_1 = cv::getOptimalNewCameraMatrix(intrinsic, distortion_coeff, test_img.size(), 1, test_img.size());
	cv::Mat new_cam_matrix_alpha_0 = cv::getOptimalNewCameraMatrix(intrinsic, distortion_coeff, test_img.size(), 0, test_img.size());
	cv::initUndistortRectifyMap(intrinsic, distortion_coeff, cv::Mat(), 
			new_cam_matrix_alpha_1, test_img.size(), CV_32FC1, map_x, map_y);
	cv::remap(test_img, u21, map_x, map_y, cv::InterpolationFlags::INTER_LINEAR,
			cv::BorderTypes::BORDER_CONSTANT, 0);
	cv::initUndistortRectifyMap(intrinsic, distortion_coeff, cv::Mat(), 
			new_cam_matrix_alpha_0, test_img.size(), CV_32FC1, map_x, map_y);
	// 矫正后效果图边上内凹,是因为对角线相对图像两侧中心有更多的视野。
	cv::remap(test_img, u22, map_x, map_y, cv::InterpolationFlags::INTER_LINEAR,
			cv::BorderTypes::BORDER_CONSTANT, 0);

	cv::Mat concat_img;
	cv::hconcat(std::vector<cv::Mat>{undistort_img2, undistort_img1}, concat_img);
	cv::imshow("undistort", concat_img);
	cv::waitKey(0);

}
// 思考
// 下面代码 得到的cImage2(畸变矫正函数输出图)与原图(有畸变图像)相同
// undistort(inImage, cImage2, intrinsic, distort_zero);  // distort_zero all 0
// 因为畸变矫正使用了distort参数计算扭曲的空间位置，而上面代码并没有使用，见文章 图像去畸原理


// remap - 去畸变和有畸变图像上点互相转换, 理解map变化

int test_remap() {
	std::string src_path = "./calib_data/left05.jpg";
	cv::Mat intrinsic = LoadParams("./workspace/intrinsic.yaml").front();
	cv::Mat distortion_coeff = LoadParams("./workspace/distortion_coeff.yaml").front();

	cv::Mat test_img = cv::imread(src_path);
	cv::Mat undistort_img;
	cv::Mat map_x, map_y;
	cv::Mat intrinsic_copy = intrinsic.clone();
	cv::initUndistortRectifyMap(intrinsic, distortion_coeff, cv::Mat(), 
			intrinsic_copy, test_img.size(), CV_32FC1, map_x, map_y);
	cv::remap(test_img, undistort_img, map_x, map_y, cv::InterpolationFlags::INTER_LINEAR,
			cv::BorderTypes::BORDER_CONSTANT, 0);

	std::vector<cv::Point2f> undistort_pts, to_un{cv::Point2f(241., 97.)};
	cv::undistortPoints(to_un, undistort_pts, intrinsic, distortion_coeff, cv::Mat(), intrinsic);
	std::cout << std::endl;
	std::cout << "undistort_pts " <<  undistort_pts.front() << std::endl;

	cv::circle(test_img, to_un.front(), 3, cv::Scalar(0,0,255));
	cv::circle(undistort_img, undistort_pts.front(), 3, cv::Scalar(0,255,0));

	cv::Mat concat_img;
	cv::hconcat(std::vector<cv::Mat>{test_img, undistort_img}, concat_img);
	// cv::imshow("undistort", concat_img);
	// cv::waitKey(0);

	/* 对mapx, mapy的理解
		dst中点的位置在src中何处去取
		pt_src = (xs, ys)
		pt_dst = (xd, yd)
		pt_dst由pt_src映射而来，是通过一特征点位置。
		xs = mapx(pt_dst)
		ys = mapy(pt_dst)
		去畸变图最左上角点坐标（238，93）, 畸变图像同一位置点(241， 97)
		mapx(238, 93) == 241, mapy(238, 93) == 97
	*/

	// 如何将获得的去畸变图像上点，映射到畸变图像上
	undistort_pts =  std::vector<cv::Point2f>{cv::Point2f(238., 93.)};
	std::vector<cv::Point2f> camera_plane_pts;	
	cv::undistortPoints(undistort_pts, camera_plane_pts, intrinsic, cv::Mat::zeros(5, 1, CV_32FC1));
	// 上句话同样效果
	// double xc = (238. - intrinsic.at<double>(0, 2)) / intrinsic.at<double>(0, 0);
	// double yc = (93. - intrinsic.at<double>(1, 2)) / intrinsic.at<double>(1, 1);
	// cv::Point3d camera_plane_pt(xc, yc, 1.);
	std::vector<cv::Point2d> distort_pts;
	cv::Point3d camera_plane_pt(camera_plane_pts.front().x, camera_plane_pts.front().y, 1.);
	cv::projectPoints(std::vector<cv::Point3d>{camera_plane_pt}, cv::Mat::zeros(3,1, CV_64FC1),
		cv::Mat::zeros(3, 1, CV_64FC1), intrinsic, distortion_coeff, distort_pts);
	return 0;
}

void test_undistortROI() {
	std::string src_path = "./calib_data/left05.jpg";
	cv::Mat intrinsic = LoadParams("./workspace/intrinsic.yaml").front();
	cv::Mat distortion_coeff = LoadParams("./workspace/distortion_coeff.yaml").front();

	cv::Mat test_img = cv::imread(src_path);
	cv::Mat undistort_img;
	cv::Mat map_x, map_y;
	cv::Mat intrinsic_roi = intrinsic.clone();
	cv::Rect roi_rect(50,50, 400, 300);
	float scale = 1.5;
	float scale_x = (float)test_img.cols / roi_rect.width;
	float scale_y = (float)test_img.rows / roi_rect.height;
	intrinsic_roi.at<double>(0, 0) = intrinsic_roi.at<double>(0, 0) * scale_x;
	intrinsic_roi.at<double>(1, 1) = intrinsic_roi.at<double>(1, 1) * scale_y;
	intrinsic_roi.at<double>(0, 2) = (intrinsic_roi.at<double>(0, 2) - roi_rect.x)* scale_y;
	intrinsic_roi.at<double>(1, 2) = (intrinsic_roi.at<double>(1, 2) - roi_rect.y) * scale_y;

	cv::initUndistortRectifyMap(intrinsic, distortion_coeff, cv::Mat(), 
			intrinsic_roi, roi_rect.size(), CV_32FC1, map_x, map_y);
	cv::remap(test_img, undistort_img, map_x, map_y, cv::InterpolationFlags::INTER_LINEAR,  
			cv::BorderTypes::BORDER_CONSTANT, 0);
	cv::imshow("src img", test_img);
	cv::imshow("undistort_img roi", undistort_img);
	cv::waitKey(0);

	std::cout << "test_undistortROI end" << std::endl;
}

int main() {
	// test_undistort();
	// test_remap();
	test_undistortROI();
	return 0;
}
