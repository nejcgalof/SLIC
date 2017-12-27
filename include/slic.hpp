using namespace std;

// Image matrix
vector<vector<double>> distances;
vector<vector<int>> labels;

vector<vector<double>> centers;
vector<vector<double>> prev_centers;
/* The number of occurences of each center. */
vector<double> center_counts;

CvPoint move_cluster_center(cv::Mat image, CvPoint center) {
	double min_grad = FLT_MAX;
	CvPoint loc_min = cvPoint(center.x, center.y);

	for (int i = center.x - 1; i < center.x + 2; i++) {
		for (int j = center.y - 1; j < center.y + 2; j++) {
			// watch gradient in gray color
			double ix_r = (int)image.at<uchar>(j + 1, i);
			double ix_l = (int)image.at<uchar>(j - 1, i);
			double iy_d = (int)image.at<uchar>(j, i + 1);
			double iy_u = (int)image.at<uchar>(j, i - 1);

			// Compute horizontal and vertical gradients and keep track of the minimum.
			// This is done to avoid centering a superpixel on an edge, and to reduce the chance of seeding a
			// superpixel with a noisy pixel
			// G(x, y) = || I(x + 1, y) − I(x − 1, y) || + || I(x, y + 1) − I(x, y − 1) ||
			// || = l2 norm
			if (sqrt(pow(ix_r - ix_l, 2)) + sqrt(pow(iy_u - iy_d, 2)) < min_grad) {
				min_grad = sqrt(pow(ix_r - ix_l, 2)) + sqrt(pow(iy_u - iy_d, 2));
				loc_min.x = i;
				loc_min.y = j;
			}
		}
	}

	return loc_min;
}

void initialize_cluster_centers(cv::Mat image, int step, int m) {
	distances.clear();
	labels.clear();
	centers.clear();
	prev_centers.clear();
	center_counts.clear();
	// cluster and distances matrix
	for (int i = 0; i < image.size().width; i++) {
		vector<int> label_row;
		vector<double> distance_row;
		for (int j = 0; j < image.size().height; j++) {
			label_row.push_back(-1);
			distance_row.push_back(FLT_MAX);
		}
		labels.push_back(label_row);
		distances.push_back(distance_row);
	}

	// Initialize cluster centers
	for (int i = step; i < image.size().width - step / 2; i += step) {
		for (int j = step; j < image.size().height - step / 2; j += step) {
			vector<double> center;
			// Find the local minimum - move cluster centers to the lowest gradient position in 3x3 neighborhood
			CvPoint nc = move_cluster_center(image, cvPoint(i, j));
			cv::Vec3b colour = image.at<cv::Vec3b>(cv::Point(nc.x, nc.y));
			
			// Generate center vector - Ck(l,a,b,x,y)
			center.push_back(colour.val[0]);
			center.push_back(colour.val[1]);
			center.push_back(colour.val[2]);
			center.push_back(nc.x);
			center.push_back(nc.y);

			// Put center information and set counter to 0
			centers.push_back(center);
			center_counts.push_back(0);
		}
	}
}

void draw_centers(cv::Mat image) {
	for (int k = 0; k < centers.size(); k++) {
		circle(image, cv::Point(centers[k][3], centers[k][4]), 11, cv::Scalar(0,0,0), CV_FILLED, 8, 0);
		circle(image, cv::Point(centers[k][3], centers[k][4]), 10, cv::Scalar(centers[k][0], centers[k][1], centers[k][2]), CV_FILLED, 8, 0);
	}
}

void display_contours(cv::Mat image, cv::Vec3b colour) {
	//neigbourhood by x and y axis
	int dx8[8] = { -1, -1,  0,  1, 1, 1, 0, -1 };
	int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1 };

	vector<CvPoint> contours;
	vector<vector<bool>> istaken; // is taken to be a contour
	for (int i = 0; i < image.size().width; i++) {
		vector<bool> nb;
		for (int j = 0; j < image.size().height; j++) {
			nb.push_back(false);
		}
		istaken.push_back(nb);
	}

	// Look all pixels
	for (int i = 0; i < image.size().width; i++) {
		for (int j = 0; j < image.size().height; j++) {
			int counter_diff_neigbours = 0;

			// Compare the pixel to 8 neighbours
			for (int k = 0; k < 8; k++) {
				int x = i + dx8[k], y = j + dy8[k];

				if (x >= 0 && x < image.size().width && y >= 0 && y < image.size().height) {
					if (istaken[x][y] == false && labels[i][j] != labels[x][y]) {
						counter_diff_neigbours += 1;
					}
				}
			}

			// If one or more different neigbours on this pixel, this pixel is part of contour
			if (counter_diff_neigbours > 1) { // How thinner line we want
				contours.push_back(cvPoint(i, j));
				istaken[i][j] = true;
			}
		}
	}

	// Draw contour pixels
	for (int i = 0; i < (int)contours.size(); i++) {
		image.at<cv::Vec3b>(cv::Point(contours[i].x, contours[i].y)) = colour;
	}
}

void colour_with_cluster_means(cv::Mat image) {
	vector<double> colours0(centers.size());
	vector<double> colours1(centers.size());
	vector<double> colours2(centers.size());
	// Get sum for all color inside contour
	for (int i = 0; i < image.size().width; i++) {
		for (int j = 0; j < image.size().height; j++) {
			int index = labels[i][j];
			cv::Vec3b colour = image.at<cv::Vec3b>(cv::Point(i, j));
			colours0[index] += colour.val[0];
			colours1[index] += colour.val[1];
			colours2[index] += colour.val[2];
		}
	}

	// Divide by the number of pixels per cluster to get the mean colour
	for (int i = 0; i < centers.size(); i++) {
		colours0[i] /= center_counts[i];
		colours1[i] /= center_counts[i];
		colours2[i] /= center_counts[i];
	}

	// Color every pixel with mean color of this cluster
	for (int i = 0; i < image.size().width; i++) {
		for (int j = 0; j < image.size().height; j++) {
			image.at<cv::Vec3b>(cv::Point(i, j)) = cv::Vec3b(colours0[labels[i][j]], colours1[labels[i][j]], colours2[labels[i][j]]);
		}
	}
}

double compute_dist(int ci, CvPoint pixel, cv::Vec3b colour, int step, int m) {
	double dc = sqrt(pow(centers[ci][0] - colour.val[0], 2) 
		+ pow(centers[ci][1] - colour.val[1], 2)
		+ pow(centers[ci][2] - colour.val[2], 2));
	double ds = sqrt(pow(centers[ci][3] - pixel.x, 2) + pow(centers[ci][4] - pixel.y, 2));

	return sqrt(pow(dc, 2) + pow(ds / step, 2) * pow(m, 2));
}

void generate_superpixels(cv::Mat image, int step, int m) {
	// Update steps 10 iterations until error converges (in paper 10 iterations) 
	bool run = true;
	int i = 0;
	while (run && i < 1000) {
		i++;
		prev_centers = centers;
	//for (int i = 0; i < 100; i++) {
		// Reset distances
		for (int j = 0; j < image.size().width; j++) {
			for (int k = 0; k < image.size().height; k++) {
				distances[j][k] = FLT_MAX;
			}
		}

		// For each cluster center
		for (int j = 0; j < centers.size(); j++) {
			// Compare each pixels in a 2 x step by 2 x step region around center
			for (int k = centers[j][3] - step; k < centers[j][3] + step; k++) {
				for (int l = centers[j][4] - step; l < centers[j][4] + step; l++) {
					if (k >= 0 && k < image.size().width && l >= 0 && l < image.size().height) {
						// Compute the distance between pixel and center 
						cv::Vec3b colour = image.at<cv::Vec3b>(cv::Point(k, l));
						double d = compute_dist(j, cvPoint(k, l), colour, step, m);

						// If this distance smaller, update then 
						if (d < distances[k][l]) {
							distances[k][l] = d;
							labels[k][l] = j;
						}
					}
				}
			}
		}

		// Clear the center values
		for (int j = 0; j < (int)centers.size(); j++) {
			centers[j][0] = centers[j][1] = centers[j][2] = centers[j][3] = centers[j][4] = 0;
			center_counts[j] = 0;
		}

		// Compute new cluster centers
		for (int j = 0; j < image.size().width; j++) {
			for (int k = 0; k < image.size().height; k++) {
				int c_id = labels[j][k];
				// If pixel have label, then update new cluster center
				if (c_id != -1) {
					cv::Vec3b colour = image.at<cv::Vec3b>(cv::Point(j, k));

					centers[c_id][0] += colour.val[0];
					centers[c_id][1] += colour.val[1];
					centers[c_id][2] += colour.val[2];
					centers[c_id][3] += j;
					centers[c_id][4] += k;

					center_counts[c_id] += 1;
				}
			}
		}

		// Normalize the clusters
		for (int j = 0; j < (int)centers.size(); j++) {
			centers[j][0] /= center_counts[j];
			centers[j][1] /= center_counts[j];
			centers[j][2] /= center_counts[j];
			centers[j][3] /= center_counts[j];
			centers[j][4] /= center_counts[j];
		}

		// Compute error
		run = false;
		for (int j = 0; j < centers.size(); j++) {
			if (0.1 < abs(prev_centers[j][3] - centers[j][3]) && 0.1 < abs(prev_centers[j][4] - centers[j][4])) {
				run = true;
			}
		}
	}
}

cv::Mat slic(cv::Mat image, int num_superpixels, int m) {
	cv::Mat lab_image = image.clone();
	cvtColor(image, lab_image, CV_BGR2Lab);
	double step = sqrt((image.cols*image.rows) / (double)num_superpixels); // Grid interval
	initialize_cluster_centers(lab_image, step, m);
	cv::Mat init_circles = lab_image.clone();
	draw_centers(init_circles);
	cvtColor(init_circles, init_circles, CV_Lab2BGR);
	imshow("Init", init_circles);
	cv::waitKey(30);
	generate_superpixels(lab_image, step, m);
	colour_with_cluster_means(image);
	display_contours(image, cv::Vec3b(0, 0, 0));
	return image;
}