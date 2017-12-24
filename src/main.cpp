#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <vector>
#include <slic.hpp>

using namespace std;

using namespace cv;

int main(int argc, char** argv )
{
    if ( argc != 3 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }

    Mat image;
    image = imread( argv[1], 1 );

    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
    namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", image);

	Mat lab_image = image.clone();
	cvtColor(image, lab_image, CV_BGR2Lab);
	int num_superpixels = atoi(argv[2]);
	double step = sqrt((image.cols*image.rows) / (double)num_superpixels); // Grid interval
	int nc = 50;
	
	initialize_cluster_centers(lab_image, step, nc);
	Mat init_circles = lab_image.clone();
	draw_centers(init_circles);
	cvtColor(init_circles, init_circles, CV_Lab2BGR);
	imshow("ak1", init_circles);
	generate_superpixels(lab_image, step, nc);
	create_connectivity(lab_image);
	colour_with_cluster_means(image);
	display_contours(image, Vec3b(0, 0, 255));
	imshow("ak", image);
    waitKey(0);
	
    return 0;
}