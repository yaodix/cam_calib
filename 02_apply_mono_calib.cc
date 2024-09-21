
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

// 利用畸变模型手写图像去畸变算法
int test_undistort_self() {


	std::string src_path = "./calib_data/left05.jpg";
	cv::Mat intrinsic = LoadParams("./workspace/intrinsic.yaml").front();
	cv::Mat distortion_coeff = LoadParams("./workspace/distortion_coeff.yaml").front();

	cv::Mat test_img = cv::imread(src_path, 0);

	//定义畸变系数
	double k1 = distortion_coeff.at<double>(0,0);
	double k2 = distortion_coeff.at<double>(1,0);
	double p1 = distortion_coeff.at<double>(2,0);
	double p2 = distortion_coeff.at<double>(3,0);
	//相机内参
	double fx = intrinsic.at<double>(0,0);
	double fy = intrinsic.at<double>(1,1);
	double cx = intrinsic.at<double>(0,2);
	double cy = intrinsic.at<double>(1,2);

	cv::Mat image_undistort = cv::Mat(test_img.rows, test_img.cols, CV_8UC1);

	//遍历每个像素，计算后去畸变
	for (int v = 0; v < test_img.rows; v++){
		for (int u = 0; u < test_img.cols; u++){
		  //根据公式计算去畸变图像上点(u, v)对应在畸变图像的坐标(u_distorted, v(distorted))，建立对应关系
			double x = (u - cx) / fx;
			double y = (v - cy) / fx;
			double r = sqrt(x * x + y * y);
			double x_distorted = x*(1+k1*r*r+k2*r*r*r*r)+2*p1*x*y+p2*(r*r+2*x*x);
			double y_distorted = y*(1+k1*r*r+k2*r*r*r*r)+2*p2*x*y+p1*(r*r+2*x*x);
			double u_distorted = fx * x_distorted + cx;
			double v_distorted = fy * y_distorted + cy;

		  //将畸变图像上点的坐标，赋值到去畸变图像中（最近邻插值）
			if (u_distorted >= 0 && v_distorted >=0 &&
          u_distorted < test_img.cols && v_distorted < test_img.rows) {
            image_undistort.at<uchar>(v, u) = test_img.at<uchar>((int)v_distorted, (int)u_distorted);
			} else {
        image_undistort.at<uchar>(v, u) = 0;
			}
		}
	}
  cv::Mat himg;
  cv::hconcat(std::vector<cv::Mat>{test_img, image_undistort}, himg);
  cv::imshow("src", himg);
	cv::waitKey(0);
}


int test_undistort() {
	std::string src_path = "./calib_data/left05.jpg";
	cv::Mat intrinsic = LoadParams("./workspace/intrinsic.yaml").front();
	cv::Mat distortion_coeff = LoadParams("./workspace/distortion_coeff.yaml").front();

	cv::Mat test_img = cv::imread(src_path);
	cv::Mat test_gray_img = cv::imread(src_path, cv::IMREAD_GRAYSCALE);

	cv::Mat undistort_img1, undistort_img2;
	// 畸变矫正方法1-undistort,最原始的畸变矫正输入参数
	cv::undistort(test_img, undistort_img1, intrinsic, distortion_coeff);

  // undistortPoints 和 undistortImagePoints 区别
  std::vector<cv::Point2f> to_un{cv::Point2f(241., 97.)}; // to_un图像中棋盘格左上角的点
  std::vector<cv::Point2f> undistort_pt1, undistort_pt2;

	cv::undistortPoints(to_un, undistort_pt1, intrinsic, distortion_coeff, cv::Mat(), intrinsic);
	cv::circle(test_img, to_un.front(), 3, cv::Scalar(0,0,255));
	cv::circle(undistort_img1, undistort_pt1.front(), 3, cv::Scalar(0,255,0));

  // opencv4.6.0+
  // cv::undistortImagePoints(to_un, undistort_pt2, intrinsic, distortion_coeff);
	// cv::circle(undistort_img1, undistort_pt2.front(), 5, cv::Scalar(0,255,0));

  cv::Mat concat_img;
	cv::hconcat(std::vector<cv::Mat>{test_img, undistort_img1}, concat_img);
	cv::imshow("undistort_src_undistort1", concat_img);

	// newCameraMatrix参数如何使用
  // 1. new camera intrinsic matrix based on the free scaling parameter.畸变矫正图像黑边大小的控制
	cv::Mat new_cam_matrix_alpha_0 = cv::getOptimalNewCameraMatrix(intrinsic, distortion_coeff, test_img.size(), 0, test_img.size());
	cv::Mat new_cam_matrix_alpha_1 = cv::getOptimalNewCameraMatrix(intrinsic, distortion_coeff, test_img.size(), 1, test_img.size());
  cv::Mat undistort_alpha0, undistort_alpha1;
  cv::undistort(test_img, undistort_alpha0, intrinsic, distortion_coeff, new_cam_matrix_alpha_0);
  cv::undistort(test_img, undistort_alpha1, intrinsic, distortion_coeff, new_cam_matrix_alpha_1);

  cv::hconcat(std::vector<cv::Mat>{undistort_alpha0, undistort_alpha1}, concat_img);
	cv::imshow("undistort_alpha_0_and_1", concat_img);

  // 2. 移动内参矩阵示例
  cv::Mat u3, u4;
	cv::Mat intrinsic_crop = intrinsic.clone();
	intrinsic_crop.at<double>(0, 2) = 200;  // 移动内参矩阵cx
	cv::undistort(test_img, u3, intrinsic, distortion_coeff, intrinsic_crop);
	// cv::undistort(test_img, u4, intrinsic_crop, distortion_coeff);  // error

  // 3. 缩放内参矩阵示例
	cv::Mat intrinsic_scale = intrinsic.clone();
	float scale = 0.5;
	intrinsic_scale.at<double>(0, 0) *= scale;
	intrinsic_scale.at<double>(0, 2) *= scale;
	intrinsic_scale.at<double>(1, 1) *= scale;
	intrinsic_scale.at<double>(1, 2) *= scale;
	cv::Mat u5, u6;
	cv::undistort(test_img, u5, intrinsic, distortion_coeff, intrinsic_scale);
	// cv::undistort(test_img, u6, intrinsic_scale, distortion_coeff); // error

	cv::hconcat(std::vector<cv::Mat>{u3, u5}, concat_img);
	cv::imshow("undistort_shift_scale", concat_img);


	cv::waitKey(0);

}


/*
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

*/


// initUndistortRectifyMap + remap 使用
int test_remap() {
	std::string src_path = "./calib_data/left05.jpg";
	cv::Mat intrinsic = LoadParams("./workspace/intrinsic.yaml").front();
	cv::Mat distortion_coeff = LoadParams("./workspace/distortion_coeff.yaml").front();

	cv::Mat test_img = cv::imread(src_path);
	cv::Mat undistort_img;
	cv::Mat map_x, map_y;
	cv::Mat intrinsic_copy = intrinsic.clone();

  cv::Mat R_30 = (cv::Mat_<double>(3,3) << 1.0000000,  0.0000000,  0.0000000,
   0.0000000,  0.9902681,  0.1391731,
   0.0000000, -0.1391731,  0.9902681 );

	cv::initUndistortRectifyMap(intrinsic, distortion_coeff, R_30, 
			intrinsic_copy, test_img.size(), CV_32FC1, map_x, map_y);
	cv::remap(test_img, undistort_img, map_x, map_y, cv::InterpolationFlags::INTER_LINEAR,
			cv::BorderTypes::BORDER_CONSTANT, 0);

	std::vector<cv::Point2f> undistort_pts, to_un{cv::Point2f(241., 97.)};
	cv::undistortPoints(to_un, undistort_pts, intrinsic, distortion_coeff, R_30, intrinsic);
	std::cout << std::endl;
	std::cout << "undistort_pts " <<  undistort_pts.front() << std::endl;

	cv::circle(test_img, to_un.front(), 3, cv::Scalar(0,0,255));
	cv::circle(undistort_img, undistort_pts.front(), 3, cv::Scalar(0,255,0));

	cv::Mat concat_img;
	cv::hconcat(std::vector<cv::Mat>{test_img, undistort_img}, concat_img);
	cv::imshow("initUndistortRectifyMap undistort", concat_img);
	cv::waitKey(0);

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
	// 以下与上句话有同样效果
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
  // test_undistort_self();
	test_undistort();
	// test_remap();
	// test_undistortROI();
	return 0;
}
