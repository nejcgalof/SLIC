#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <vector>
#include <experimental/filesystem>
#include <slic.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv )
{
	if (argc == 3) {

		Mat image;
		image = imread(argv[1], 1);
		if (!image.data)
		{
			printf("No image data \n");
			return -1;
		}
		imshow("Input Image", image);
		waitKey(30);
		int m = atoi(argv[2]);
		VideoWriter video(std::experimental::filesystem::path(argv[1]).stem().string() + "_" + std::string(argv[2]) + ".avi", CV_FOURCC('M', 'J', 'P', 'G'), 1, Size(image.size().width, image.size().height), true);
		video.write(image);
		for (int i = 10; i < 2000; i+=20) {
			Mat result_slic = slic(image.clone(), i, m);
			imshow("Result SLIC", result_slic);
			video.write(result_slic);
			waitKey(30);
		}

		return 0;
		
	}
    if ( argc != 4 )
    {
        printf("usage: Slic.exe <Image_Path> <Superpixel> <m-weight> \n");
        return -1;
    }

    Mat image;
    image = imread( argv[1], 1 );

    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
    imshow("Input Image", image);
	waitKey(30);
	int num_superpixels = atoi(argv[2]);
	int m = atoi(argv[3]);
	Mat result_slic=slic(image, num_superpixels, m);
	imshow("Result SLIC", result_slic);
	imwrite(std::experimental::filesystem::path(argv[1]).stem().string() + "_" + std::string(argv[2]) + "_" + string(argv[3]) + ".jpg", result_slic);
    waitKey(0);
	
    return 0;
}