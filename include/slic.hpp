using namespace std;

// Image matrix
vector<vector<double>> distances;
vector<vector<int>> labels;

vector<vector<double>> centers;
/* The number of occurences of each center. */
vector<int> center_counts;

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

void initialize_cluster_centers(cv::Mat image, int step, int nc) {
	// cluster and distances matrix
	for (int i = 0; i < image.size().width; i++) {
		vector<int> label_row;
		vector<double> distance_row;
		for (int j = 0; j < image.size().width; j++) {
			label_row.push_back(-1);
			distance_row.push_back(INT_MAX);
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