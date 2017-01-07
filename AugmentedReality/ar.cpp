#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>

using namespace cv;
using namespace std;
using namespace cv::aruco;


int main()
{
	VideoCapture cap(1);
	
	Mat image;

	namedWindow("Video");
	namedWindow("Marker");
	Mat markerImage;

	Ptr<Dictionary> dictionary = getPredefinedDictionary(DICT_6X6_1000);
	Ptr<DetectorParameters> parameters = DetectorParameters::create();
	drawMarker(dictionary, 23, 300, markerImage, 1);

	imshow("Marker", markerImage);

	FileStorage fs;
	fs.open("camera.xml", FileStorage::READ);

	Mat intrinsic;
	Mat distCoeffs;

	fs["Intrinsic"] >> intrinsic;
	fs["DistortionCoefficients"] >> distCoeffs;

	fs.release();

	while(true)
	{
		vector<int> markerIds;
		vector<vector<Point2f>> markerCorners, rejectedCandidatees;
		cap >> image;
		detectMarkers(image, dictionary, markerCorners, markerIds, parameters, rejectedCandidatees);
		drawDetectedMarkers(image, markerCorners, markerIds);

		if (markerIds.size() > 0)
		{

			vector<Vec3d> r_vecs, t_vecs;
			estimatePoseSingleMarkers(markerCorners, 0.05, intrinsic, distCoeffs, r_vecs, t_vecs);
			drawAxis(image, intrinsic, distCoeffs, r_vecs, t_vecs, 0.1);
		}
		imshow("Video", image);
		waitKey(1);
	}


	waitKey(0);

	return 0;

}