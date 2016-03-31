/********************************************************************************/
/*                  ChanVese Segmentation Algorithm Header                      */
/*   Based on Project Paper Image Segmentation Using the Chan-Vese Algorithm    */
/*          By: Robert Crandall   ECE 532 Project  Fall, 2009                   */
/*        Implemented By : Kirankumar Adam (kiranadam@gmail.com)                */
/********************************************************************************/

#ifndef CHANVESE_HPP
#define CHANVESE_HPP

#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

class ChanVese
{
	private :
	
		// Parameters as in section 4 of the paper
		double lambda1;
		double lambda2;
		int p;
		double mu;
		double nu;
		double h;
		double dt;
		
		// functions as needed
		vector<double> Heavyside_avg(Mat& src, Mat& phi, double h);
		Mat delta_phi(Mat& phi);
		double perimeter_length(Mat& phi, Mat& del_phi);
		vector<double> CV_coefficients(Mat& phi, Mat& del_phi, double L, unsigned i, unsigned j);
		Mat ReinitPhi(Mat& phi, unsigned iterations);		
		
		
	public :
		// Constructors 
		ChanVese();
		ChanVese(double lambda1, double lambda2, int p, double mu, double nu, double h, double dt);

		// Segmentation 
		Mat ChanVese_segment(Mat& src, Mat& phi0, unsigned iterations);

};

#endif
