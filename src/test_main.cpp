#include "ChanVese.hpp"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <omp.h>
#include <algorithm>

using namespace std;
using namespace cv;

int main(int, char**)
{
	Mat src = imread("../../ChanVese/img_test.jpg",0); // input gray scale image
  
	Mat phi0 = Mat::zeros(src.rows,src.cols,CV_64FC1);
  	Mat phi, edge;

	double lambda1 = 1.02;
	double lambda2 = 1.0;
	int p = 1;
	double mu = 70;
	double nu = 1900;
	double h = 1;
	double dt = 0.1;


  	// Create object of ChanVese class
	ChanVese c(lambda1,lambda2,p,mu,nu,h,dt);

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
      			phi0.ptr<double>(i)[j] = 900.0/(900.0 + x*x + y*y) - 0.5;
    		}
  	}

  	// Segmentation
  	phi = c.ChanVese_segment(src,phi0, 2);
	
	Mat U_phi;
	phi.convertTo(U_phi,CV_8UC1);

	imwrite("../../ChanVese/output.jpg",U_phi);
	imshow("Phi",U_phi);
	waitKey(-1);
	cout<<"Done Segmentation"<<endl;
  
	/*
	// edges
  	edge = c.edge_representation(phi);
	imshow("Edges",edge);

  	imwrite("CV_Egde.jpg",edge);

	Mat output = Mat::zeros(src.rows,src.cols,CV_8UC1);
  
  	for(unsigned i = 0; i < src.rows; i++)
  	{
    		for(unsigned j = 0; j < src.cols; j++)
    		{
      			if(phi.at<double>(i,j) >= 0)
			{
        			output.at<uchar>(i,j) = (uchar)0;
			}
      			else
			{
        			output.at<uchar>(i,j) = (uchar)255;
			}
    		}
  	}

  	imwrite("CV_Regions.pgm",output);
	
	*/
	return 0;
}
