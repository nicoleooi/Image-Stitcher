//Written by Nicole Ooi and Connor Chappelle
#include <iostream>
#include "opencv2/opencv.hpp"
#include <sstream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/video/background_segm.hpp>
#include <cmath>
#include <tuple>
#include <sstream>

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

using namespace cv;
using namespace std;

vector<KeyPoint> detectMe(Mat img) {
	Ptr<FeatureDetector> detector = ORB::create(1000, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31);
	vector <KeyPoint> kp;
	detector->detect(img, kp);
	return kp;
}

Mat describeMe(Mat img, vector<KeyPoint> kp) {
	Ptr<DescriptorExtractor> descriptor = ORB::create(1000, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31);
	Mat desc;
	descriptor->compute(img, kp, desc);
	return desc;
}

vector<DMatch> matchMe(Mat img1, Mat img2) {
	vector<KeyPoint> kp1 = detectMe(img1);
	vector<KeyPoint> kp2 = detectMe(img2);

	Mat desc1 = describeMe(img1, kp1);
	Mat desc2 = describeMe(img2, kp2);

	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
	vector<DMatch> temp_matches; //temporary matches
	vector<DMatch> matches; //matches once we've filtered by our float thingy
	float ratio = 0.7;
	matcher->match(desc1, desc2, temp_matches);

	//filters the temp matches before pushing into our array of matches
	for (int i = 0; i < (temp_matches.size() - 1); i++) {
		if (temp_matches[i].distance < (ratio * temp_matches[i + 1].distance)) {
			matches.push_back(temp_matches[i]);
		}
	}
	return matches;
}

Mat stitcher(Mat img1, Mat img2) { 
	//1. img1 should be passed in WITH a border
	//2. match it! 
	//3. show the matches
	//4. warp img2 to match img1

	vector<KeyPoint> kp1 = detectMe(img1);
	vector<KeyPoint> kp2 = detectMe(img2);
	Mat desc1 = describeMe(img1, kp1);
	Mat desc2 = describeMe(img2, kp2);
	cout << "done describing and detecting" << endl;

	vector<DMatch> matches = matchMe(img1, img2);
	//extract for homography
	vector<Point2f> p1, p2;
	for (int i = 0; i < (int)matches.size(); i++){
		p1.push_back(kp1[matches[i].queryIdx].pt);
		p2.push_back(kp2[matches[i].trainIdx].pt);
	}

	cout << "matches done" << endl;
	Mat out;
	drawMatches(img1, kp1, img2, kp2, matches, out);

	Mat H = findHomography(p1, p2, RANSAC);
	
	cout << H << endl;
	Mat result(img1.rows, img1.cols, CV_8UC3);
	warpPerspective(img2, result, H.inv(), result.size());
	namedWindow("ah", WINDOW_NORMAL);
	imshow("ah", result);
	waitKey();

	cout << "Image 1 size: " << img1.size() << endl;
	cout << "Result size: " << result.size() << endl;

	//this is to get the shape of the transformed one, so we can make 
	//a poly
	vector<Point2f> borderSrc, borderTrans;
	borderSrc.push_back(Point(0, 0));
	borderSrc.push_back(Point(0, img2.rows));
	borderSrc.push_back(Point(img2.cols, 0));
	borderSrc.push_back(Point(img2.cols, img2.rows));
	perspectiveTransform(borderSrc, borderTrans, H.inv());

	vector<Point> newBorder;
	newBorder.push_back(Point((int)borderTrans[0].x, (int)borderTrans[0].y));
	newBorder.push_back(Point((int)borderTrans[1].x, (int)borderTrans[1].y));
	newBorder.push_back(Point((int)borderTrans[3].x, (int)borderTrans[3].y));
	newBorder.push_back(Point((int)borderTrans[2].x, (int)borderTrans[2].y));

	//merged will hold the flat image, with the shape of the warped image on top
	Mat merged = img1.clone();
	fillConvexPoly(merged, newBorder, Scalar(0, 0, 0));
	namedWindow("shape", WINDOW_NORMAL);
	imshow("shape", merged);
	waitKey();

	//at this point, they should have the same size but double check here
	cout << "pic w shape: " << merged.size() << endl;
	cout << "homography pic" << result.size() << endl;

	//loop through the one w the shape (merged)
	Mat output(merged.rows, merged.cols, CV_8UC3);
	bitwise_or(merged, result, output);


	return output;
}

void pairwise(String folder) {
	int t = 185;
	if (folder.find("StJames") > -1 ) {
		t = 185;
	}
	else if (folder.find("WLH") > -1) {
		t = 100;
	}
	else if (folder.find("office2" > -1)) {
		t = 200;
	}
	else {
		cout << "Please rename the folder to StJames, WLH, or office2 !" << endl;
	}

	//---------STEP 1-------------
	vector<String> filenames;
	vector<Mat> images;
	vector<vector<KeyPoint>> keypoints;
	vector<Mat> descriptors;
	glob(folder, filenames);
	for (int i = 0; i < filenames.size(); i++) {
		Mat tmp = imread(filenames[i]);
		resize(tmp, tmp, Size(tmp.cols / 3, tmp.rows / 3));
		vector<KeyPoint> kp = detectMe(tmp);
		Mat desc = describeMe(tmp, kp);
		keypoints.push_back(kp);
		descriptors.push_back(desc);
		images.push_back(tmp);
	}
	cout << "Done computing kps, descs" << endl;

	vector<Mat> pairs;
	int i = 0;
	//----------STEP 3-----------
	while (1) { 
		i++;
		if (images.size() <= 1) {
			break;
		}
		Mat seed = images[0];
		copyMakeBorder(seed, seed, 2 * seed.cols, 2 * seed.cols, 2 * seed.cols, 2 * seed.cols, BORDER_CONSTANT, 0);
		namedWindow("pair 1", WINDOW_NORMAL);
		imshow("pair 1", seed);
		waitKey(); //stjames thresh = 185
		int best_idx = 0;

		vector<DMatch> matches;
		//recalc seed kps, descs
		Mat seed_bw;
		cvtColor(seed, seed_bw, COLOR_BGR2GRAY);
		vector<KeyPoint> seed_kp = detectMe(seed_bw);
		Mat seed_desc = describeMe(seed_bw, seed_kp);

		//for each other image,
		//calculate matches according to the seed image
		//if matches > t
		for (int i = 1; i < images.size(); i++) {
			Mat tmp;
			cvtColor(images[i], tmp, COLOR_BGR2GRAY);

			matches = matchMe(seed_bw, tmp);
			//stop using the black n whites
			cout << matches.size() << endl;

			if (matches.size() > t) {
				best_idx = i;
				break;
			}
		}
		namedWindow("pair 2", WINDOW_NORMAL);
		imshow("pair 2", images[best_idx]);
		waitKey();

		if (best_idx == 0) { 
			cout << "No good match was found" << endl;
			images.erase(images.begin()); //scrap bad matches
			continue;
		}

		cout << "Determined best match at image " << best_idx << endl;

		Mat output = stitcher(seed, images[best_idx]);
		namedWindow("Out", WINDOW_NORMAL);
		imshow("Out", output);
		waitKey();
		
		std::ostringstream name;
		name << folder << "_" << i << ".jpg";
		imwrite(name.str(), output);
		
		GaussianBlur(output, output, Size(3, 3), 0, 0);

		resize(output, output, Size(output.cols / 3, output.rows / 3));
		pairs.push_back(output);

		images.erase(images.begin() + best_idx);
		images.erase(images.begin());
	}

	cout << "Pairs is complete" << endl;

	//now we have our pairs
	//match our pairs with each other
	Mat p = pairs[0];

	while (1) {
		if (pairs.size() <= 1) {
			break;
		}
		namedWindow("pair 1", WINDOW_NORMAL);
		imshow("pair 1", p);
		waitKey(); //stjames thresh = 185
		int best_idx = 0;

		vector<DMatch> matches;
		//recalc kps, descs on the merged one
		Mat p_bw;
		cvtColor(p, p_bw, COLOR_BGR2GRAY);
		vector<KeyPoint> p_kp = detectMe(p_bw);
		Mat p_desc = describeMe(p_bw, p_kp);

		//get the best match for your merged one and merge it
		for (int i = 1; i < pairs.size(); i++) {
			Mat tmp;
			cvtColor(pairs[i], tmp, COLOR_BGR2GRAY);

			matches = matchMe(p_bw, tmp);
			//stop using the black n whites
			cout << matches.size() << endl;

			if (matches.size() > t) {//we r easy just take it
				best_idx = i;
				break;
			}
		}

		namedWindow("pair 2", WINDOW_NORMAL);
		imshow("pair 2", pairs[best_idx]);
		waitKey();

		if (best_idx == 0) {
			cout << "No good match was found" << endl;
			continue;
		}

		cout << "Determined best match at pair " << best_idx << endl;

		Mat output = stitcher(p, pairs[best_idx]);
		namedWindow("Out", WINDOW_NORMAL);
		imshow("Out", output);
		waitKey();

		pairs.erase(pairs.begin());
		pairs.erase(pairs.begin() + best_idx);
		pairs.push_back(output);
	}
}

void main() {
	
	pairwise("StJames");
	pairwise("WLH");
	pairwise("office2");
}
