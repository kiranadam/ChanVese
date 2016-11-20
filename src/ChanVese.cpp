/********************************************************************************/
/*                  ChanVese Segmentation Algorithm C++ file                    */
/*   Based on Project Paper Image Segmentation Using the Chan-Vese Algorithm    */
/*          By: Robert Crandall   ECE 532 Project  Fall, 2009                   */
/*        Implemented By : Kirankumar Adam (kiranadam@gmail.com)                */
/********************************************************************************/

#include "ChanVese.hpp"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <cmath>
#include <omp.h>
#include <algorithm>
#include <vector>

using namespace std;
using namespace cv;

ChanVese :: ChanVese()
{
	this->lambda1 = 1.0;
	this->lambda2 = 1.0;
	this->p = 1;
	this->mu = 0.5;
	this->nu = 0.0;
	this->h = 1;
	this->dt = 0.1;
}

ChanVese :: ChanVese(double lambda1, double lambda2, int p, double mu, double nu, double h, double dt)
{
	this->lambda1 = lambda1;
	this->lambda2 = lambda2;
	this->p = p;
	this->mu = mu;
	this->nu = nu;
	this->h = h;
	this->dt = dt;
}

// Assuming for 2D case smooth step 
vector<double> ChanVese :: Heavyside_avg(Mat& src, Mat& phi, double epsilon)
{
	// Check section III from paper (Active Contours Without Edges by Chan and Vese from IEEE TRANSACTIONS ON IMAGE PROCESSING, VOL. 10, NO. 2, FEBRUARY 2001) for heeavyside function

	double c1_num = 0;
	double c1_den = 0;
	double c2_num = 0;
	double c2_den = 0;
	
	for(unsigned i=0; i<src.rows; i++)
	{
		#pragma omp parallel for
		for(unsigned j=0; j<src.cols; j++)
		{
			double H_phi = 0.5*(1+(2.0/CV_PI)*atan2((double)phi.ptr<uchar>(i)[j],epsilon));
			c1_num += (double)src.ptr<uchar>(i)[j]*H_phi;
			c1_den += H_phi;
			c2_num += (double)src.ptr<uchar>(i)[j]*(1-H_phi);
			c2_den += 1-H_phi;
		}
	}

	double c1_phi = c1_num/c1_den;   // check equation (6) from the paper.
	double c2_phi = c2_num/c2_den;   // check equation (7) from the paper.	
	
	vector<double> c_phi;

	c_phi.push_back(c1_phi);
	c_phi.push_back(c2_phi);

	return c_phi;
}

Mat ChanVese :: delta_phi(Mat& phi)
{
	Mat del_phi = Mat::zeros(phi.rows, phi.cols, CV_64FC1);

	for(unsigned i=0; i<phi.rows; i++)
	{
		#pragma omp parallel for
		for(unsigned j=0; j<phi.cols; j++)
		{
			del_phi.ptr<double>(i)[j] = (1/CV_PI)*(h/(pow(h,2) + pow((double)phi.ptr<uchar>(i)[j],2)));
		}
	}

	return del_phi;
}

double ChanVese :: perimeter_length(Mat& phi, Mat& del_phi)
{
	Mat nabla_phi, abs_nabla_phi;
	
	// The Nabla operator calculation	
	Laplacian(phi, nabla_phi, CV_64F, 3, 1, 0, BORDER_DEFAULT);
    	convertScaleAbs(nabla_phi, abs_nabla_phi);

	double L = 0;
	
	for(unsigned i=0; i<phi.rows; i++)
	{
		#pragma omp parallel for
		for(unsigned j=0; j<phi.cols; j++)
		{
			L += del_phi.ptr<double>(i)[j] * abs_nabla_phi.ptr<double>(i)[j];
		}
	}

	return L;
}

vector<double> ChanVese :: CV_coefficients(Mat& phi, Mat& del_phi, double L, unsigned i, unsigned j)
{
	vector<double> F;

	// defining constants
	double C[4];
	double zero_nullifier = 0.0001; // constant to avoid divide by zero case

	C[0] = 1/sqrt(pow((phi.ptr<double>(i+1)[j]-phi.ptr<double>(i)[j]),2) + pow((phi.ptr<double>(i)[j+1]-phi.ptr<double>(i)[j-1]),2)/4 + zero_nullifier);
	C[1] = 1/sqrt(pow((phi.ptr<double>(i)[j]-phi.ptr<double>(i-1)[j]),2) + pow((phi.ptr<double>(i-1)[j+1]-phi.ptr<double>(i-1)[j-1]),2)/4 + zero_nullifier);
	C[2] = 1/sqrt(pow((phi.ptr<double>(i+1)[j]-phi.ptr<double>(i-1)[j]),2)/4 + pow((phi.ptr<double>(i)[j+1]-phi.ptr<double>(i)[j]),2) + zero_nullifier);
	C[3] = 1/sqrt(pow((phi.ptr<double>(i+1)[j-1]-phi.ptr<double>(i-1)[j-1]),2)/4 + pow((phi.ptr<double>(i)[j]-phi.ptr<double>(i)[j-1]),2) + zero_nullifier); 

	double F_den = h + dt*del_phi.ptr<double>(i)[j] * mu * (p*pow(L,p-1)) * (C[0]+C[1]+C[2]+C[3]);
	double F_num = 0.0;

	for(unsigned k=0; k<4; k++)
	{
		F_num = (dt * del_phi.ptr<double>(i)[j] * mu *(p*pow(L,p-1)) * C[k]) / F_den;
		F.push_back(F_num);
	}

	F_num = h / F_den;

	F.push_back(F_num);

	return F;
}

Mat ChanVese :: ChanVese_segment(Mat& src, Mat& phi0, unsigned iteration)
{
	vector<double> c;
	Mat del_phi;
	double L;
	vector<double> F;

	//initialize phi based on phi0
	Mat phi = phi0.clone();
	Mat new_phi = Mat::zeros(src.rows, src.cols, CV_64FC1);

	for(unsigned k=0; k<iteration; k++)
	{
		// calculate c1 and c2 
		c = Heavyside_avg(src, phi, h);
		 
		for(unsigned l=0; l<iteration; l++)
		{
			// calculate delta_phi
			del_phi = delta_phi(phi);

			if(p == 1)
			{
				L = 1.0;
			}
			else
			{
				L = perimeter_length(phi, del_phi);
			}

			for(unsigned i=1; i < src.rows-1; i++)
			{
				#pragma omp parallel for
				for(unsigned j=1; j < src.cols-1; j++)
				{
					F = CV_coefficients( phi, del_phi, L, i, j);
				
					double Pi_j = phi.ptr<double>(i)[j] + dt * del_phi.ptr<double>(i)[j]*(nu + lambda1*pow(((double)src.ptr<uchar>(i)[j]-c[0]),2) + lambda2*pow(((double)src.ptr<uchar>(i)[j]-c[1]),2) );  
					//Update phi		
					new_phi.ptr<double>(i)[j] = F[0]*phi.ptr<double>(i+1)[j] + F[1]*phi.ptr<double>(i-1)[j] + F[2]*phi.ptr<double>(i)[j+1] + F[3]*phi.ptr<double>(i)[j-1] + F[4]*Pi_j;
				}
			}

			// Border update 
			for(unsigned i=0; i < src.rows; i++)
			{
				new_phi.ptr<double>(i)[0] = new_phi.ptr<double>(i)[1];
				new_phi.ptr<double>(i)[src.cols-1] = new_phi.ptr<double>(i)[src.cols-2];
			}

			for(unsigned j=0; j < src.cols; j++)
			{
				new_phi.ptr<double>(0)[j] = new_phi.ptr<double>(1)[j];
				new_phi.ptr<double>(src.rows-1)[j] = new_phi.ptr<double>(src.rows-2)[j];
			}
			
			new_phi = ReinitPhi(new_phi,100);
			phi = new_phi.clone();
		}
	}
	return new_phi;
}


Mat ChanVese :: ReinitPhi(Mat& phi, unsigned iterations)
{
 	Mat psi = phi.clone();

  	double a;
  	double b;
  	double c;
  	double d;
  	double x;
  	double G;

  	bool flag = false;
  	double Q;

  	unsigned M;

  	Mat old_psi;

  	for(unsigned k = 0; k < iterations && flag == false; k++)
  	{
    		old_psi = psi.clone();

    		for(unsigned i = 1; i < phi.rows-1; i++)
    		{
      			for(unsigned j = 1; j < phi.cols-1; j++)
     			{
        				a = (phi.ptr<double>(i)[j] - phi.ptr<double>(i-1)[j])/h;
        				b = (phi.ptr<double>(i+1)[j] - phi.ptr<double>(i)[j])/h;
        				c = (phi.ptr<double>(i)[j] - phi.ptr<double>(i)[j-1])/h;
        				d = (phi.ptr<double>(i)[j+1] - phi.ptr<double>(i)[j])/h;

        				if(phi.ptr<double>(i)[j] > 0)
				{
          				G = sqrt(max(pow(max(a,0.0),2),pow(min(b,0.0),2)) + max(pow(max(c,0.0),2),pow(min(d,0.0),2))) - 1.0;
				}
        				else if(phi.ptr<double>(i)[j] < 0)
				{
          				G = sqrt(max(pow(min(a,0.0),2),pow(max(b,0.0),2)) + max(pow(min(c,0.0),2),pow(max(d,0.0),2))) - 1.0;
				}
        				else
				{
          				G = 0;
				}

        				x = (phi.ptr<double>(i)[j]= 0)?(1.0):(-1.0);
        				psi.ptr<double>(i)[j] = psi.ptr<double>(i)[j] - dt*x*G;
      			}
    		}

    		// Checking condition for stop
    		Q = 0.0;
		M = 0.0;

    		for(unsigned i = 0; i < phi.rows; i++)
    		{
			#pragma omp parallel for
      			for(unsigned j = 0; j < phi.cols; j++)
      			{
        				if(abs(old_psi.ptr<double>(i)[j]) <= h)
        				{
          				M++;
          				Q += abs(old_psi.ptr<double>(i)[j] - psi.ptr<double>(i)[j]);
        				}
      			}
    		}

    		if (M != 0)
		{
      			Q /= (double)M;
		}
    		else
		{
      			Q = 0.0;
		}

    		if (Q < dt*h*h)
    		{
      			flag = true;
    		}
    
  	}

	return psi;
}



Mat ChanVese :: edge_representation(Mat& phi)
{
  
	Mat edge = Mat::zeros(phi.rows,phi.cols,CV_8UC1);

  	for(unsigned i = 0; i < phi.rows; i++)
  	{
    		for(unsigned j = 0; j < phi.cols; j++)
    		{
      			// Checking interior pixels to avoid bounds checking
      			if (i > 0 && i < phi.rows-1 && j > 0 && j < phi.cols-1)
      			{
        				if(phi.ptr<double>(i)[j] == 0)
        				{
          				if(phi.ptr<double>(i-1)[j-1] != 0 || phi.ptr<double>(i-1)[j] != 0 ||
           				   phi.ptr<double>(i-1)[j+1] != 0 || phi.ptr<double>(i)[j-1] != 0 ||
           				   phi.ptr<double>(i)[j+1] != 0 || phi.ptr<double>(i+1)[j-1] != 0 ||
           				   phi.ptr<double>(i+1)[j] != 0 || phi.ptr<double>(i+1)[j+1] != 0 )
					{
             					edge.ptr<uchar>(i)[j] = (uchar)255;
           				}
        				}
        				else
        				{
          				if(abs(phi.ptr<double>(i)[j]) < abs(phi.ptr<double>(i-1)[j-1]) && (phi.ptr<double>(i)[j]>0) != (phi.ptr<double>(i-1)[j-1]>0))
					{
             					edge.ptr<uchar>(i)[j] = (uchar)255;
					}
     					else if(abs(phi.ptr<double>(i)[j]) < abs(phi.ptr<double>(i-1)[j]) && (phi.ptr<double>(i)[j]>0) != (phi.ptr<double>(i-1)[j]>0))
					{
             					edge.ptr<uchar>(i)[j] = (uchar)255;
					}
     					else if(abs(phi.ptr<double>(i)[j]) < abs(phi.ptr<double>(i-1)[j+1]) && (phi.ptr<double>(i)[j]>0) != (phi.ptr<double>(i-1)[j+1]>0))
					{
             					edge.ptr<uchar>(i)[j] = (uchar)255;
					}
     					else if(abs(phi.ptr<double>(i)[j]) < abs(phi.ptr<double>(i)[j-1]) && (phi.ptr<double>(i)[j]>0) != (phi.ptr<double>(i)[j-1]>0))
					{
             					edge.ptr<uchar>(i)[j] = (uchar)255;
					}
     					else if(abs(phi.ptr<double>(i)[j]) < abs(phi.ptr<double>(i)[j+1]) && (phi.ptr<double>(i)[j]>0) != (phi.ptr<double>(i)[j+1]>0))
					{
       						edge.ptr<uchar>(i)[j] = (uchar)255;
					}
     					else if(abs(phi.ptr<double>(i)[j]) < abs(phi.ptr<double>(i+1)[j-1]) && (phi.ptr<double>(i)[j]>0) != (phi.ptr<double>(i+1)[j-1]>0))
					{
             					edge.ptr<uchar>(i)[j] = (uchar)255;
					}
     					else if(abs(phi.ptr<double>(i)[j]) < abs(phi.ptr<double>(i+1)[j]) && (phi.ptr<double>(i)[j]>0) != (phi.ptr<double>(i+1)[j]>0))
					{
       						edge.ptr<uchar>(i)[j] = (uchar)255;
					}
     					else if(abs(phi.ptr<double>(i)[j]) < abs(phi.ptr<double>(i+1)[j+1]) && (phi.ptr<double>(i)[j]>0) != (phi.ptr<double>(i+1)[j+1]>0))
       					{  
						edge.ptr<uchar>(i)[j] = (uchar)255;
        					}
      				}
    			}
  		}
	}
}
