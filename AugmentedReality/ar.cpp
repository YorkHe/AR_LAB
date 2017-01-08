
#define GLEW_STATIC
#include <GL/glew.h>

#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>

#include "Camera.hpp"
#include "Shader.h"
#include "Model.hpp"

using namespace cv;
using namespace std;
using namespace cv::aruco;

const int markersX = 6;
const int markersY = 4;

int windowWidth = 1024;
int windowHeight = 768;

GLFWwindow* window;

const char* ARWindowName = "Augmented Reality";

GLuint bgTexture;

GLfloat bgVertices[] = {
	// Positions       // Texture Coords
	 1.0f,  1.0f, 0.0f, 1.0f, 1.0f,   // Top Right
	 1.0f, -1.0f, 0.0f, 1.0f, 0.0f,   // Bottom Right
	-1.0f, -1.0f, 0.0f, 0.0f, 0.0f,   // Bottom Left
	-1.0f,  1.0f, 0.0f, 0.0f, 1.0f    // Top Left 
};

GLuint bgIndices[] = {
	0, 1, 3,
	1, 2, 3
};

float borderColor[] = { 1.0f, 1.0f, 0.0f, 1.0f };

GLuint bgVAO, bgVBO, bgEBO;

Shader bgShader;


static void cGetBoardObjectAndImagePoints(const Ptr<Board> &_board, InputArray _detectedIds,
	InputArrayOfArrays _detectedCorners,
	OutputArray _imgPoints, OutputArray _objPoints) {

	CV_Assert(_board->ids.size() == _board->objPoints.size());
	CV_Assert(_detectedIds.total() == _detectedCorners.total());

	size_t nDetectedMarkers = _detectedIds.total();

	vector< Point3f > objPnts;
	objPnts.reserve(nDetectedMarkers);

	vector< Point2f > imgPnts;
	imgPnts.reserve(nDetectedMarkers);

	// look for detected markers that belong to the board and get their information
	for (unsigned int i = 0; i < nDetectedMarkers; i++) {
		int currentId = _detectedIds.getMat().ptr< int >(0)[i];
		if (currentId >= markersX * markersY)
			continue;
		for (unsigned int j = 0; j < _board->ids.size(); j++) {
			if (currentId == _board->ids[j]) {
				for (int p = 0; p < 4; p++) {
					objPnts.push_back(_board->objPoints[j][p]);
					imgPnts.push_back(_detectedCorners.getMat(i).ptr< Point2f >(0)[p]);
				}
			}
		}
	}

	// create output
	Mat(objPnts).copyTo(_objPoints);
	Mat(imgPnts).copyTo(_imgPoints);
}

int cEstimatePoseBoard(InputArrayOfArrays _corners, InputArray _ids, const Ptr<Board> &board,
	InputArray _cameraMatrix, InputArray _distCoeffs, OutputArray _rvec,
	OutputArray _tvec) {

	CV_Assert(_corners.total() == _ids.total());

	// get object and image points for the solvePnP function
	Mat objPoints, imgPoints;
	cGetBoardObjectAndImagePoints(board, _ids, _corners, imgPoints, objPoints);

	CV_Assert(imgPoints.total() == objPoints.total());

	if (objPoints.total() == 0) // 0 of the detected markers in board
		return 0;

	bool useExtrinsicGuess = true;
	if (_rvec.empty() || _tvec.empty())
	{
		_rvec.create(3, 1, CV_64FC1);
		_tvec.create(3, 1, CV_64FC1);
		useExtrinsicGuess = false;
	}

	solvePnP(objPoints, imgPoints, _cameraMatrix, _distCoeffs, _rvec, _tvec, useExtrinsicGuess, CV_EPNP);

	cout << _rvec.getMat() << endl;
	cout << _tvec.getMat() << endl;
	// divide by four since all the four corners are concatenated in the array for each marker
	return (int)objPoints.total() / 4;
}

int initGLEnv()
{
	glfwInit();

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

	window = glfwCreateWindow(windowWidth, windowHeight, ARWindowName, nullptr, nullptr);
	glfwMakeContextCurrent(window);
	glewExperimental = GL_TRUE;

	GLenum err;
	if ((err = glewInit()) != GLEW_OK)
	{
		cerr << "Failed to initalize GLEW" << endl;
		cerr << glewGetErrorString(err) << endl;
		glfwTerminate();

		return -1;
	}

	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_DEPTH_TEST);

	glGenTextures(1, &bgTexture);

	glGenVertexArrays(1, &bgVAO);
	glGenBuffers(1, &bgVBO);
	glGenBuffers(1, &bgEBO);

	glBindVertexArray(bgVAO);

	glBindBuffer(GL_ARRAY_BUFFER, bgVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(bgVertices), bgVertices, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bgEBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(bgIndices), bgIndices, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (GLvoid*)0);
	glEnableVertexAttribArray(0);

	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));
	glEnableVertexAttribArray(1);

	glBindVertexArray(0);
	
	return 0;
}
void drawBackground(InputArray image)
{
	Mat _image = image.getMat();
	flip(_image, _image, 0);
	cvtColor(_image, _image, CV_BGR2RGB);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, bgTexture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
	glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	bgShader.Use();
	glUniform1i(glGetUniformLocation(bgShader.Program, "bgImage"), 0);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, _image.cols, _image.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, _image.data);
//	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, _image.cols, _image.rows, GL_RGB, GL_UNSIGNED_BYTE, _image.data);

	glBindVertexArray(bgVAO);
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
	glBindVertexArray(0);
}
void drawScene(InputArray image)
{
	glClearColor(0.5f, 0.5f, 0.5f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


	drawBackground(image);

	glfwSwapBuffers(window);
}

int main()
{

	VideoCapture cap(1);

	
	Mat image;

	/*
	namedWindow(ARWindowName, WINDOW_OPENGL);
	resizeWindow(ARWindowName, windowWidth, windowHeight);
	setOpenGlContext(ARWindowName);
	*/

	initGLEnv();

	bgShader = Shader("bg_v.glsl", "bg_f.glsl");
	Model statue("libertyStatue/LibertStatue.obj");
	

	namedWindow("Marker");


	int markerLength = 150;
	int markerSeparation = 25;

	Size imageSize;
	imageSize.width = markersX * (markerLength + markerSeparation) - markerSeparation + 2 * markerSeparation;
	imageSize.height = markersY * (markerLength + markerSeparation) - markerSeparation + 2 * markerSeparation;

	Ptr<Dictionary> dictionary = getPredefinedDictionary(DICT_7X7_1000);

	Ptr<DetectorParameters> parameters = DetectorParameters::create();

	Ptr<GridBoard> board = GridBoard::create(markersX, markersY, float(markerLength), float(markerSeparation), dictionary);

	Mat boardImage;
	board->draw(imageSize, boardImage, markerSeparation, 1);



	Mat markerImage1;
	Mat markerImage2;

	drawMarker(dictionary, 100, 200, markerImage1, 1);
	drawMarker(dictionary, 200, 200, markerImage2, 1);

	imwrite("marker.jpg", boardImage);
	imwrite("marker_left.jpg", markerImage1);
	imwrite("marker_right.jpg", markerImage2);

	imshow("Marker", boardImage);

	FileStorage fs;
	fs.open("camera.xml", FileStorage::READ);

	Mat intrinsic;
	Mat distCoeffs;

	fs["Intrinsic"] >> intrinsic;
	fs["DistortionCoefficients"] >> distCoeffs;

	fs.release();

	while(!glfwWindowShouldClose(window))
	{
		vector<int> markerIds;
		vector<vector<Point2f>> markerCorners, rejectedCandidatees;
		cap >> image;
		detectMarkers(image, dictionary, markerCorners, markerIds, parameters, rejectedCandidatees);
		refineDetectedMarkers(image, board, markerCorners, markerIds, rejectedCandidatees, intrinsic, distCoeffs);
		drawDetectedMarkers(image, markerCorners, markerIds);
		if (markerIds.size() > 0)
		{

			Vec3d r_vecs, t_vecs;
			int markers = cEstimatePoseBoard(markerCorners, markerIds, board, intrinsic, distCoeffs, r_vecs, t_vecs);

			if (markers > 0)
				drawAxis(image, intrinsic, distCoeffs, r_vecs, t_vecs, 100);

			vector<vector<Point2f>> controllerCorners;
			for (int i = 0; i < markerIds.size(); i++)
			{
				switch (markerIds[i])
				{
				case 100:
					controllerCorners.push_back(markerCorners[i]);
					break;
				case 200:
					controllerCorners.push_back(markerCorners[i]);
					break;
				default:
					break;
				}
			}

			vector<Vec3d> c_r_vecs;
			vector<Vec3d> c_t_vecs;

			estimatePoseSingleMarkers(controllerCorners, 10, intrinsic, distCoeffs, c_r_vecs, c_t_vecs);

			for (int i = 0; i < c_r_vecs.size(); i++)
				drawAxis(image, intrinsic, distCoeffs, c_r_vecs[i], c_t_vecs[i], 5);
		}
		drawScene(image);
		waitKey(1);
	}


	waitKey(0);

	return 0;

}