// pnp问题
// head pose estimation

#include <opencv2/opencv.hpp>
 
using namespace std;
using namespace cv;

const int boardWidth = 9;								 // 横向的角点数目
const int boardHeight = 6;								// 纵向的角点数据
const int squareSize = 25;								// 标定板黑白格子的大小，单位 mm
const Size boardSize = Size(boardWidth, boardHeight);

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

void test_pnp() {
    Mat rvec, tvec;
	std::vector<cv::Point3f> world_points;
	for (int rowIndex = 0; rowIndex < boardHeight; rowIndex++) { // boardheight=6;
		for (int colIndex = 0; colIndex < boardWidth; colIndex++) { // boardWidth = 9;		
			world_points.push_back(cv::Point3f(rowIndex * squareSize, colIndex * squareSize, 0));  
		}
	}
    cv::Mat intrinsic = LoadParams("./workspace/intrinsic.yaml").front();
	cv::Mat distortion_coeff = LoadParams("./workspace/distortion_coeff.yaml").front();

    std::string src_path = "./calib_data/left05.jpg";
	cv::Mat test_gray_img = cv::imread(src_path, cv::IMREAD_GRAYSCALE);
    std::vector<cv::Point2f> corner;									// 某一副图像找到的角点
    bool isFind = findChessboardCorners(test_gray_img, boardSize, corner);

	solvePnP(world_points, corner, intrinsic, distortion_coeff, rvec, tvec);
	cv::Mat R;
	cv::Rodrigues(rvec, R); // R is 3x3

    std::cout << "rvec " << rvec <<  std::endl;
    std::cout << "tvec " << tvec <<  std::endl;
    std::cout << "R " << R <<  std::endl;

}
 
int main(int argc, char **argv) {
    // test_pnp();
    // return 0;
    // Read input image
 
    // 2D image points. If you change the image, you need to change vector
    std::vector<cv::Point2d> image_points;
    std::vector<cv::Point3d> model_points;
    // left img
    // cv::Mat im = cv::imread("./workspace/headPose_lett.jpg");
    // image_points.push_back( cv::Point2d(359, 391) );    // Nose tip
    // image_points.push_back( cv::Point2d(399, 561) );    // Chin
    // image_points.push_back( cv::Point2d(337, 297) );     // Left eye left corner
    // image_points.push_back( cv::Point2d(513, 301) );    // Right eye right corner
    // image_points.push_back( cv::Point2d(345, 465) );    // Left Mouth corner
    // image_points.push_back( cv::Point2d(453, 469) );    // Right mouth corner

    // right img
    cv::Mat im = cv::imread("./workspace/headPose_right.jpg");
    image_points.push_back( cv::Point2d(839, 391) );    // Nose tip
    image_points.push_back( cv::Point2d(806, 561) );    // Chin
    image_points.push_back( cv::Point2d(861, 297) );    // Left eye left corner
    image_points.push_back( cv::Point2d(688, 301) );    // Right eye right corner
    image_points.push_back( cv::Point2d(850, 465) );    // Left Mouth corner
    image_points.push_back( cv::Point2d(741, 469) );    // Right mouth corner
 
    // 3D model points.
    model_points.push_back(cv::Point3d(0.0f, 0.0f, 0.0f));               // Nose tip
    model_points.push_back(cv::Point3d(0.0f, -330.0f, -65.0f));          // Chin
    model_points.push_back(cv::Point3d(-225.0f, 170.0f, -135.0f));       // Left eye left corner
    model_points.push_back(cv::Point3d(225.0f, 170.0f, -135.0f));        // Right eye right corner
    model_points.push_back(cv::Point3d(-150.0f, -150.0f, -125.0f));      // Left Mouth corner
    model_points.push_back(cv::Point3d(150.0f, -150.0f, -125.0f));       // Right mouth corner
 
    // Camera internals
    double focal_length = im.cols; // Approximate focal length.
    Point2d center = cv::Point2d(im.cols/2,im.rows/2);
    cv::Mat camera_matrix = (cv::Mat_<double>(3,3) << focal_length, 0, center.x, 0 , focal_length, center.y, 0, 0, 1);
    cv::Mat dist_coeffs = cv::Mat::zeros(5,1,cv::DataType<double>::type); // Assuming no lens distortion
 
    cout << "Camera Matrix " << endl << camera_matrix << endl ;
    // Output rotation and translation
    cv::Mat rotation_vector; // Rotation in axis-angle form
    cv::Mat translation_vector;
 
    // Solve for pose, solvePnPRansac better than solvePnp
    cv::solvePnPRansac(model_points, image_points, camera_matrix, dist_coeffs, rotation_vector, translation_vector);
 
    // Project a 3D point (0, 0, 1000.0) onto the image plane. 
 
    vector<Point3d> nose_end_point3D;
    vector<Point2d> nose_end_point2D;
    nose_end_point3D.push_back(Point3d(0,0,500.0));
 
    projectPoints(nose_end_point3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs, nose_end_point2D);
 
    for(int i=0; i < image_points.size(); i++) {
        circle(im, image_points[i], 3, Scalar(0,0,255), -1);
    }
 
    cv::line(im, image_points[0], nose_end_point2D[0], cv::Scalar(255,0,0), 2);
 
    cout << "Rotation Vector " << endl << rotation_vector << endl;
    cout << "Translation Vector" << endl << translation_vector << endl;
 
    cout <<  nose_end_point2D << endl;
 
    // Display image.
    cv::imshow("Output", im);
    cv::waitKey(0); 
}


