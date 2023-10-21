// 人脸朝向计算


// pnp问题
	Mat rvec, tvec;
	int index = 2;
	solvePnP(objRealPoint[index], distance_corners, intrinsic, distortion_coeff, rvec, tvec);
	cv::Mat R;
	cv::Rodrigues(rvec, R); // R is 3x3

	//R = R.t();  // rotation of inverse
	//tvec = -R * tvec; // translation of inverse,相机坐标原点在世界坐标系位置
