//============================================================================
// Name        : Aia2.cpp
// Author      : Ronny Haensch
// Version     : 1.0
// Copyright   : -
// Description : 
//============================================================================

#include "Aia2.h"

// calculates the contour line of all objects in an image
/*
img			the input image
objList		vector of contours, each represented by a two-channel matrix
thresh		threshold used to binarize the image
k			number of applications of the erosion operator
*/
void Aia2::getContourLine(const Mat& img, vector<Mat>& objList, int thresh, int k){
    
    //binarization the gray picture
    Mat img_bin;
    threshold(img, img_bin, thresh, 1, CV_THRESH_BINARY_INV );
    
    //erosion the binary image
    Mat element = getStructuringElement(MORPH_RECT, Size(3,3));
    erode(img_bin,img_bin,element,Point(-1,-1),k);
    
    //find out the input image's contours
    vector<vector<Point>> contours;
    findContours( img_bin, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE );
    
    for (int i = 0; i < contours.size(); i++) {
        Mat new_contour = Mat(contours.at(i),true);
        objList.push_back(new_contour);
    }
}

// calculates the (unnormalized!) fourier descriptor from a list of points
/*
contour		1xN 2-channel matrix, containing N points (x in first, y in second channel)
out		fourier descriptor (not normalized)
*/
Mat Aia2::makeFD(const Mat& contour){
    
    Mat float_contour;
    contour.convertTo(float_contour, CV_32FC2);

    Mat complexI;
    dft(float_contour, complexI, DFT_SCALE|DFT_REAL_OUTPUT);
    
    return complexI;
}

// normalize a given fourier descriptor
/*
fd		the given fourier descriptor
n		number of used frequencies (should be even)
out		the normalized fourier descriptor
*/
Mat Aia2::normFD(const Mat& fd, int n){
  
  plotFD(fd, "fd not normalized", 0);

  // translation invariance
  Mat planes[] = {Mat::zeros(fd.size(), CV_32F),Mat::zeros(fd.size(), CV_32F)};
  split(fd, planes);
  planes[0].at<float>(0,0) = 0.0f;
  planes[1].at<float>(0,0) = 0.0f;
  Mat normTrans = Mat::zeros(fd.size(), CV_32FC2);
  merge(planes,2,normTrans);
  plotFD(normTrans, "fd translation invariant", 0);
  
  // scale invariance
  Mat magnitude, angle;
  cartToPolar(planes[0],planes[1],magnitude,angle);
  // get F(1)
  float F1 = magnitude.at<float>(0,1);
  // get F(-1)
  float FN1 = magnitude.at<float>(magnitude.rows-1,magnitude.cols-1);
  cout<<magnitude<<endl;
  if (0.0f != max(F1, FN1)) {
      divide(magnitude, Scalar(max(F1,FN1)), magnitude);
  }
  polarToCart(magnitude, angle, planes[0], planes[1]);
    
  Mat mv[] = {Mat::zeros(planes[0].size(), CV_32F),Mat::zeros(planes[1].size(), CV_32F)};
  multiply(planes[0], Scalar(50), mv[0]);
  multiply(planes[1], Scalar(50), mv[1]);
  Mat normScale;
  merge(mv,2,normScale);
  plotFD(normScale, "fd translation and scale invariant", 0);
  
  // rotation invariance
  float rotation = (angle.at<float>(0,1) + angle.at<float>(angle.rows-1,angle.cols-1) )/2;
  add(angle, Scalar(rotation), angle);
  polarToCart(magnitude, angle, planes[0], planes[1]);
  multiply(planes[0], Scalar(50), mv[0]);
  multiply(planes[1], Scalar(50), mv[1]);
  Mat normRotat;
  merge(mv,2,normRotat);
  plotFD(normRotat, "fd translation, scale, and rotation invariant", 0);
  
  // smaller sensitivity for details
  int counter = 0;
  Mat smaller_details[] = {Mat::zeros(n,1, CV_32F),Mat::zeros(n,1, CV_32F)};
  for (int row = 0; row < planes[0].rows; row++) {
      for ( int col = 0; col < planes[0].cols; col++) {
          if ( counter < n/2 ) {
              smaller_details[0].at<float>(row,0) = planes[0].at<float>(row,col);
              smaller_details[0].at<float>(n-row-1,0) = planes[0].at<float>(planes[0].rows-1-row,planes[0].cols-1-col);
              counter++;
          }
      }
  }
  
  counter = 0;
  for (int row = 0; row < planes[1].rows; row++) {
      for ( int col = 0; col < planes[1].cols; col++) {
          if ( counter < n/2 ) {
              smaller_details[1].at<float>(row,0) = planes[1].at<float>(row,col);
              smaller_details[1].at<float>(n-row-1,0) = planes[1].at<float>(planes[0].rows-1-row,planes[0].cols-1-col);
              counter++;
          }
      }
  }
  multiply(smaller_details[0], Scalar(50), mv[0]);
  multiply(smaller_details[1], Scalar(50), mv[1]);
  Mat normSmaller;
  merge(mv,2,normSmaller);
  plotFD(normSmaller, "fd translation, scale, and rotation invariant, smaller sensitivity", 0);

  magnitude = Mat::zeros(fd.rows, fd.cols, CV_32F);
  angle = Mat::zeros(fd.rows, fd.cols, CV_32F);
  cartToPolar(smaller_details[0],smaller_details[1],magnitude,angle);
  return magnitude;
}

// plot fourier descriptor
/*
fd	the fourier descriptor to be displayed
win	the window name
dur	wait number of ms or until key is pressed
*/
void Aia2::plotFD(const Mat& fd, string win, double dur){
    Mat ifd = Mat::zeros(fd.size(), CV_32FC2);
    dft(fd,ifd,DFT_INVERSE);
    
    Mat planes[] = {Mat::zeros(fd.size(), CV_32F),
                    Mat::zeros(fd.size(), CV_32F)};
   
    split(ifd, planes); //planes[0] Re, planes[1] Im
    
    double Re_min, Re_max;
    cv::minMaxLoc(planes[0], &Re_min, &Re_max);
    double Im_min, Im_max;
    cv::minMaxLoc(planes[1], &Im_min, &Im_max);
    
    add(planes[0], Scalar(abs(Re_min)), planes[0]);
    add(planes[1], Scalar(abs(Im_min)), planes[1]);
    
    vector<vector<Point>> contours;
    vector<Point> contour;
    for (int row = 0; row < planes[0].rows; row++) {
        for (int col = 0; col < planes[0].cols; col++) {
            int x = (int)planes[0].at<float>(row,col);
            int y = (int)planes[1].at<float>(row,col);
            Point point = Point(x,y);
            contour.push_back(point);
        }
    }
    contours.push_back(contour);
    
    cv::minMaxLoc(planes[0], &Re_min, &Re_max);
    cv::minMaxLoc(planes[1], &Im_min, &Im_max);
    
    cv::Mat fdImage(Size((int)Re_max,(int)Im_max), CV_8U, cv::Scalar(0,0,0));
    cv::Scalar color;
    color = cv::Scalar(255, 255, 255);
    drawContours(fdImage, contours, 0, color);
    
    namedWindow( win.c_str() );
    imshow( win.c_str(), fdImage );
    imwrite( win +".jpg", fdImage);
    waitKey(dur);
}

/* *****************************
  GIVEN FUNCTIONS
***************************** */

// function loads input image, calls processing functions, and saves result
// in particular extracts FDs and compares them to templates
/*
img			path to query image
template1	path to template image of class 1
template2	path to template image of class 2
*/
void Aia2::run(string img, string template1, string template2){

	// process image data base
	// load image as gray-scale, paths in argv[2] and argv[3]
	Mat exC1 = imread( template1, 0);
	Mat exC2  = imread( template2, 0);
	if ( (!exC1.data) || (!exC2.data) ){
	    cout << "ERROR: Cannot load class examples in\n" << template1 << "\n" << template2 << endl;
	    cerr << "Press enter to continue..." << endl;
	    cin.get();
	    exit(-1);
	}

	// parameters
	// these two will be adjusted below for each image indiviudally
	int binThreshold;				// threshold for image binarization
	int numOfErosions;				// number of applications of the erosion operator
	// these two values work fine, but might be interesting for you to play around with them
	int steps = 32;					// number of dimensions of the FD
	double detThreshold = 0.01;		// threshold for detection

	// get contour line from images
	vector<Mat> contourLines1;
	vector<Mat> contourLines2;
	// --> Adjust threshold and number of erosion operations
	binThreshold = 150;
	numOfErosions = 4;
	getContourLine(exC1, contourLines1, binThreshold, numOfErosions);
	int mSize = 0, mc1 = 0, mc2 = 0, i = 0;
	for(vector<Mat>::iterator c = contourLines1.begin(); c != contourLines1.end(); c++,i++){
		if (mSize<c->rows){
			mSize = c->rows;
			mc1 = i;
		}
	}
	getContourLine(exC2, contourLines2, binThreshold, numOfErosions);
	for(vector<Mat>::iterator c = contourLines2.begin(); c != contourLines2.end(); c++, i++){
		if (mSize<c->rows){
			mSize = c->rows;
			mc2 = i;
		}
	}
	// calculate fourier descriptor
	Mat fd1 = makeFD(contourLines1.at(mc1));
	Mat fd2 = makeFD(contourLines2.at(mc2));

	// normalize  fourier descriptor
	Mat fd1_norm = normFD(fd1, steps);
	Mat fd2_norm = normFD(fd2, steps);

	// process query image
	// load image as gray-scale, path in argv[1]
	Mat query = imread( img, 0);
	if (!query.data){
	    cerr << "ERROR: Cannot load query image in\n" << img << endl;
	    cerr << "Press enter to continue..." << endl;
	    cin.get();
	    exit(-1);
	}
	
	// get contour lines from image
	vector<Mat> contourLines;
	// --> Adjust threshold and number of erosion operations
	binThreshold = 140;
	numOfErosions = 4;
	getContourLine(query, contourLines, binThreshold, numOfErosions);
	
	cout << "Found " << contourLines.size() << " object candidates" << endl;

	// just to visualize classification result
	Mat result(query.rows, query.cols, CV_8UC3);
	vector<Mat> tmp;
	tmp.push_back(query);
	tmp.push_back(query);
	tmp.push_back(query);
	merge(tmp, result);

	// loop through all contours found
	i = 1;
	for(vector<Mat>::iterator c = contourLines.begin(); c != contourLines.end(); c++, i++){

	    cout << "Checking object candidate no " << i << " :\t";
	  
		// color current object in yellow
	  	Vec3b col(0,255,255);
	    for(int p=0; p < c->rows; p++){
			result.at<Vec3b>(c->at<Vec2i>(p)[1], c->at<Vec2i>(p)[0]) = col;
	    }
	    showImage(result, "result", 0);
	    
	    // if fourier descriptor has too few components (too small contour), then skip it (and color it in blue)
	    if (c->rows < steps){
			cout << "Too less boundary points (" << c->rows << " instead of " << steps << ")" << endl;
			col = Vec3b(255,0,0);
	    }else{
			// calculate fourier descriptor
			Mat fd = makeFD(*c);
			// normalize fourier descriptor
			Mat fd_norm = normFD(fd, steps);
			// compare fourier descriptors
			double err1 = norm(fd_norm, fd1_norm)/steps;
			double err2 = norm(fd_norm, fd2_norm)/steps;
			// if similarity is too small, then reject (and color in cyan)
			if (min(err1, err2) > detThreshold){
				cout << "No class instance ( " << min(err1, err2) << " )" << endl;
				col = Vec3b(255,255,0);
			}else{
				// otherwise: assign color according to class
				if (err1 > err2){
					col = Vec3b(0,0,255);
					cout << "Class 2 ( " << err2 << " )" << endl;
				}else{
					col = Vec3b(0,255,0);
					cout << "Class 1 ( " << err1 << " )" << endl;
				}
			}
		}
		// draw detection result
	    for(int p=0; p < c->rows; p++){
			result.at<Vec3b>(c->at<Vec2i>(p)[1], c->at<Vec2i>(p)[0]) = col;
	    }
	    // for intermediate results, use the following line
	    showImage(result, "result", 0);

	}
	// save result
	imwrite("result.png", result);
	// show final result
	showImage(result, "result", 0);
}

// shows the image
/*
img	the image to be displayed
win	the window name
dur	wait number of ms or until key is pressed
*/
void Aia2::showImage(const Mat& img, string win, double dur){
  
    // use copy for normalization
    Mat tempDisplay = img.clone();
    if (img.channels() == 1) normalize(img, tempDisplay, 0, 255, CV_MINMAX);
    // create window and display omage
    namedWindow( win.c_str(), CV_WINDOW_AUTOSIZE );
    imshow( win.c_str(), tempDisplay );
    // wait
    if (dur>=0) waitKey(dur);
    
}

// function loads input image and calls processing function
// output is tested on "correctness" 
void Aia2::test(void){
	
	//test_getContourLine();
	//test_makeFD();
	//test_normFD();
	
}

void Aia2::test_getContourLine(void){

	vector<Mat> objList;
	Mat img(100, 100, CV_8UC1, Scalar(255));
	Mat roi(img, Rect(40,40,20,20));
	roi.setTo(0);
	getContourLine(img, objList, 128, 1);
	Mat cline(68,1,CV_32SC2);
	int k=0;
	for(int i=41; i<58; i++) cline.at<Vec2i>(k++) = Vec2i(41,i);
	for(int i=41; i<58; i++) cline.at<Vec2i>(k++) = Vec2i(i,58);
	for(int i=58; i>41; i--) cline.at<Vec2i>(k++) = Vec2i(58, i);
	for(int i=58; i>41; i--) cline.at<Vec2i>(k++) = Vec2i(i,41);
	if ( sum(cline != objList.at(0)).val[0] != 0 ){
		cout << "There might be a problem with Aia2::getContourLine(..)!" << endl;
		cin.get();
	}
}

void Aia2::test_makeFD(void){

	Mat cline(68,1,CV_32SC2);
	int k=0;
	for(int i=41; i<58; i++) cline.at<Vec2i>(k++) = Vec2i(41,i);
	for(int i=41; i<58; i++) cline.at<Vec2i>(k++) = Vec2i(i,58);
	for(int i=58; i>41; i--) cline.at<Vec2i>(k++) = Vec2i(58, i);
	for(int i=58; i>41; i--) cline.at<Vec2i>(k++) = Vec2i(i,41);
	
	Mat fd = makeFD(cline);
	if (fd.rows != cline.rows){
		cout << "There is be a problem with Aia2::makeFD(..):" << endl;
		cout << "\tThe number of frequencies does not match the number of contour points" << endl;
		cin.get();
		exit(-1);
	}
	if (fd.channels() != 2){
		cout << "There is be a problem with Aia2::makeFD(..):" << endl;
		cout << "\tThe fourier descriptor is supposed to be a two-channel, 1D matrix" << endl;
		cin.get();
		exit(-1);
	}
}

void Aia2::test_normFD(void){

	double eps = pow(10,-3);

	Mat cline(68,1,CV_32SC2);
	int k=0;
	for(int i=41; i<58; i++) cline.at<Vec2i>(k++) = Vec2i(41,i);
	for(int i=41; i<58; i++) cline.at<Vec2i>(k++) = Vec2i(i,58);
	for(int i=58; i>41; i--) cline.at<Vec2i>(k++) = Vec2i(58, i);
	for(int i=58; i>41; i--) cline.at<Vec2i>(k++) = Vec2i(i,41);
	
	Mat fd = makeFD(cline);
	Mat nfd = normFD(fd, 32);
	if (nfd.channels() != 1){
		cout << "There is be a problem with Aia2::normFD(..):" << endl;
		cout << "\tThe normalized fourier descriptor is supposed to be a one-channel, 1D matrix" << endl;
		cin.get();
		exit(-1);
	}
	if (abs(nfd.at<float>(0)) > eps){
		cout << "There is be a problem with Aia2::normFD(..):" << endl;
		cout << "\tThe F(0)-component of the normalized fourier descriptor F is supposed to be 0" << endl;
		cin.get();
		exit(-1);
	}
	if ((abs(nfd.at<float>(1)-1.) > eps) && (abs(nfd.at<float>(nfd.rows-1)-1.) > eps)){
		cout << "There is be a problem with Aia2::normFD(..):" << endl;
		cout << "\tThe F(1)-component of the normalized fourier descriptor F is supposed to be 1" << endl;
		cout << "\tBut what if the unnormalized F(1)=0?" << endl;
		cin.get();
		exit(-1);
	}
	if (nfd.rows != 32){
		cout << "There is be a problem with Aia2::normFD(..):" << endl;
		cout << "\tThe number of components does not match the specified number of components" << endl;
		cin.get();
		exit(-1);
	}
}
