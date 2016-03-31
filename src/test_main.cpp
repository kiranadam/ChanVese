
#include "ChanVese.hpp"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <omp.h>
#include <algorithm>

using namespace std;
using namespace cv;

int main(int, char**)
{
	Mat src = imread("../../ChanVese/M-78B-1400_2.jpg",0); // input gray scale image
  
	Mat phi0 = Mat::zeros(src.rows,src.cols,CV_64FC1);
  	Mat phi, edge;


  	// Create object of ChanVese class
	ChanVese c;

  	// Set up initial circular contour 
  	double x;
  	double y;
  	for(unsigned i = 0; i < src.rows; i++)
  	{
		#pragma omp parallel for
    		for(unsigned j = 0; j < src.cols; j++)
    		{
      			x = double(i) - src.rows/2.0;
      			y = double(j) - src.cols/2.0;
      			phi0.at<double>(i,j) = 900.0/(900.0 + x*x + y*y) - 0.5;
    		}
  	}

  	// Segmentation
  	phi = c.ChanVese_segment(src,phi0, 5);
	imshow("Phi",phi);
	waitKey(-1);
	cout<<"Done Segmentation"<<endl;
  
	return 0;
}
