#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

using namespace std;
using namespace cv;
using namespace cv::ml;

void segmentate1(vector<Mat> images, vector<String> fileNames, string setName, vector<String> classes, Ptr<SVM> svm, Scalar mean, Scalar stdDev);
void segmentate2(vector<Mat> images, vector<String> fileNames, string setName, vector<String> classes, Ptr<SVM> svm, Scalar mean, Scalar stdDev, vector<int*> coordinates);
void avgSaturation(Mat source, Mat& result);
void cannySeg(Mat source, Mat& result);
void findLargestContour(Mat source, Mat& noFillResult, Mat& filledResult);
void removeSmallContours(Mat source, Mat& result, double threshold);
void boundingBox(Mat source, Mat& result, Rect& rect);
void colourSegHSV(Mat source, Mat& result);
void FeatureExtraction(Mat image, Mat& features);

#define SEPH 3
#define SEPV 15
void createWindowPartition(Mat srcI, Mat& largeWin, Mat win[], Mat legends[], int noOfImagePerCol = 1,
	int noOfImagePerRow = 1, int sepH = SEPH, int sepV = SEPV);

Mat kernel = (Mat_<unsigned char>(3, 3)
	<< 1, 1, 1,
	1, 1, 1,
	1, 1, 1);

int main()
{
	Ptr<SVM> SVM = SVM::load("Trained models\\SVM - all rows.xml");
	ifstream readNorm("Trained models\\Norm - all rows.txt");
	double meantemp, stdDevtemp;
	readNorm >> meantemp;
	readNorm >> stdDevtemp;
	readNorm.close();
	Scalar mean(meantemp), stdDev(stdDevtemp);
	ifstream readClasses("Trained models\\Labels - all rows.txt");
	vector<String> classes = {};
	String className;
	while (getline(readClasses, className)) {
		classes.push_back(className);
	}

	auto start = chrono::high_resolution_clock::now();
	// set A folder location: Inputs/SetA
	system("dir \"Inputs\\SetA\\*.png\" /b > dirList.txt");
	ifstream setA("dirList.txt");
	vector<Mat> imagesA = {};
	vector<String> fileNamesA = {};
	string fileNameA;
	while (getline(setA, fileNameA)) {
		imagesA.push_back(imread("Inputs/SetA/" + fileNameA));
		fileNamesA.push_back(fileNameA);
	}
	setA.close();
	segmentate1(imagesA, fileNamesA, "setA", classes, SVM, mean, stdDev);

	// set B folder location: Inputs/SetB
	system("dir \"Inputs\\SetB\\*.png\" /b > dirList.txt");
	ifstream setB("dirList.txt");
	vector<Mat> imagesB = {};
	vector<String> fileNamesB = {};
	string fileNameB;
	while (getline(setB, fileNameB)) {
		imagesB.push_back(imread("Inputs/SetB/" + fileNameB));
		fileNamesB.push_back(fileNameB);
	}
	setB.close();
	segmentate1(imagesB, fileNamesB, "setB", classes, SVM, mean, stdDev);

	// set C folder location: Inputs/SetC
	system("dir \"Inputs\\SetC\\*.png\" /b > dirList.txt");
	ifstream setC("dirList.txt");
	vector<Mat> imagesC = {};
	vector<String> fileNamesC = {};
	string fileNameC;
	while (getline(setC, fileNameC)) {
		imagesC.push_back(imread("Inputs/SetC/" + fileNameC));
		fileNamesC.push_back(fileNameC);
	}
	setC.close();

	ifstream readCoordinates("Inputs\\SetC\\Coordinations.txt");
	vector<int*> coordinates = {};
	for (int i = 0; i < imagesC.size(); i++) {
		int* coordinate = new int[4];
		readCoordinates >> coordinate[0];
		readCoordinates >> coordinate[1];
		readCoordinates >> coordinate[2];
		readCoordinates >> coordinate[3];
		coordinates.push_back(coordinate);
	}

	segmentate2(imagesC, fileNamesC, "setC", classes, SVM, mean, stdDev, coordinates);

	// set D folder location: Inputs/SetD
	system("dir \"Inputs\\SetD\\*.png\" /b > dirList.txt");
	ifstream setD("dirList.txt");
	vector<Mat> imagesD = {};
	vector<String> fileNamesD = {};
	string fileNameD;
	while (getline(setD, fileNameD)) {
		imagesD.push_back(imread("Inputs/SetD/" + fileNameD));
		fileNamesD.push_back(fileNameD);
	}
	setD.close();
	segmentate1(imagesD, fileNamesD, "setD", classes, SVM, mean, stdDev);
	
	system("del dirList.txt");
	auto stop = chrono::high_resolution_clock::now();
	auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start).count();
	cout << "Time taken " << duration << "ms" << endl;
	int totalImage = imagesA.size() + imagesB.size() + imagesC.size() + imagesD.size();
	cout << "Total image = " << totalImage << endl;
	cout << "Time taken per image = " << duration / totalImage << "ms" << endl;
	return 0;
}

void segmentate1(vector<Mat> images, vector<String> fileNames, string setName, vector<String> classes, Ptr<SVM> svm, Scalar mean, Scalar stdDev) {
	Size outputSize = Size(150, 150);
	Mat results;
	const int noOfImagePerCol = 100, noOfImagePerRow = 3;
	Mat win[noOfImagePerCol * noOfImagePerRow];
	Mat legends[noOfImagePerCol * noOfImagePerRow];
	int totalResultImageCount = 0;
	int correctPrediction = 0;
	cout << setName << " prediction result" << endl;
	for (int i = 0; i < images.size(); i++) {
		int currentIndex = i % 100;
		if (currentIndex == 0) {
			createWindowPartition(Mat(outputSize, CV_8UC3), results, win, legends, noOfImagePerCol, noOfImagePerRow);
		}

		// show original image
		resize(images[i], win[currentIndex * noOfImagePerRow], outputSize, INTER_LINEAR);
		putText(legends[currentIndex * noOfImagePerRow], "[1]: Original Image", Point(1, 12), 1, .7, Scalar(255, 255, 255), 1);

		// segment out more saturated regions
		Mat hsvMask;
		colourSegHSV(images[i], hsvMask);
		resize(hsvMask, win[currentIndex * noOfImagePerRow + 1], outputSize, INTER_LINEAR);
		putText(legends[currentIndex * noOfImagePerRow + 1], "[2]: HSV mask", Point(1, 12), 1, .7, Scalar(255, 255, 255), 1);

		// draw bounding box
		Mat finalMask;
		Rect rect;
		images[i].copyTo(finalMask);
		boundingBox(hsvMask, finalMask, rect);
		resize(finalMask, win[currentIndex * noOfImagePerRow + 2], outputSize, INTER_LINEAR);
		putText(legends[currentIndex * noOfImagePerRow + 2], "[3]: Bounding box", Point(1, 12), 1, .7, Scalar(255, 255, 255), 1);

		if (currentIndex == 99 || i == images.size() - 1) {
			imwrite(setName + " results " + to_string(totalResultImageCount++) + ".png", results(Range(0, outputSize.height * currentIndex + SEPV * currentIndex), Range::all()));
		}

		Mat feature;
		FeatureExtraction(images[i], feature);
		subtract(feature, mean, feature);
		divide(feature, stdDev, feature);
		String actualLabel = fileNames[i].substr(0, fileNames[i].size() - 4);
		int actualLabelIndex = 0;
		for (int j = 0; j < classes.size(); j++) {
			if (actualLabel == classes[j]) {
				actualLabelIndex = j;
			}
		}
		int predictedLabelIndex = svm->predict(feature);
		String predictedLabel = classes[predictedLabelIndex];
		if (actualLabelIndex == predictedLabelIndex) {
			correctPrediction++;
		}
		cout << "Actual: " << actualLabelIndex << "\tPredicted: " << predictedLabelIndex << "\t(" << actualLabel << " as " << predictedLabel << ")" << endl;
	}
	cout << "Recognition Accuracy: " << correctPrediction / (double)images.size() << endl;
	cout << endl << endl;
}

void segmentate2(vector<Mat> images, vector<String> fileNames, string setName, vector<String> classes, Ptr<SVM> svm, Scalar mean, Scalar stdDev, vector<int*> coordinates) {
	Size outputSize = Size(150, 150);
	Mat results;
	const int noOfImagePerCol = 100, noOfImagePerRow = 11;
	Mat win[noOfImagePerCol * noOfImagePerRow];
	Mat legends[noOfImagePerCol * noOfImagePerRow];
	int totalResultImageCount = 0;
	int correctPrediction = 0;
	double total_accuracy = 0;
	cout << setName << " prediction result" << endl;
	for (int i = 0; i < images.size(); i++) {
		int currentIndex = i % 100;
		if (currentIndex == 0) {
			createWindowPartition(Mat(outputSize, CV_8UC3), results, win, legends, noOfImagePerCol, noOfImagePerRow);
		}

		// show original image
		resize(images[i], win[currentIndex * noOfImagePerRow], outputSize, INTER_LINEAR);
		putText(legends[currentIndex * noOfImagePerRow], "[1]: Original Image", Point(1, 12), 1, .7, Scalar(255, 255, 255), 1);

		// segment out more saturated regions
		Mat saturationMask;
		avgSaturation(images[i], saturationMask);
		resize(saturationMask, win[currentIndex * noOfImagePerRow + 1], outputSize, INTER_LINEAR);
		putText(legends[currentIndex * noOfImagePerRow + 1], "[2]: HSV mask", Point(1, 12), 1, .7, Scalar(255, 255, 255), 1);

		// canny edge based on hue and saturation
		Mat cannyMask, dilatedCannyMask;
		cannySeg(images[i], cannyMask);
		dilate(cannyMask, dilatedCannyMask, kernel);
		resize(cannyMask, win[currentIndex * noOfImagePerRow + 2], outputSize, INTER_LINEAR);
		putText(legends[currentIndex * noOfImagePerRow + 2], "[3]: Canny mask", Point(1, 12), 1, .7, Scalar(255, 255, 255), 1);

		// find longest edge
		Mat edgeLargestContourMaskNoFill, edgeLargestContourMaskFilled;
		findLargestContour(dilatedCannyMask, edgeLargestContourMaskNoFill, edgeLargestContourMaskFilled);
		resize(edgeLargestContourMaskFilled, win[currentIndex * noOfImagePerRow + 3], outputSize, INTER_LINEAR);
		putText(legends[currentIndex * noOfImagePerRow + 3], "[4]: Filled longest contour in [3]", Point(1, 12), 1, .7, Scalar(255, 255, 255), 1);

		// saturation mask & filled longest contour formed by canny edge
		Mat satAndFilled = saturationMask & edgeLargestContourMaskFilled;
		resize(satAndFilled, win[currentIndex * noOfImagePerRow + 4], outputSize, INTER_LINEAR);
		putText(legends[currentIndex * noOfImagePerRow + 4], "[5]: [2] & [4]", Point(1, 12), 1, .7, Scalar(255, 255, 255), 1);

		// cut the mask found with edge
		Mat cutWithEdge = satAndFilled - cannyMask;
		resize(cutWithEdge, win[currentIndex * noOfImagePerRow + 5], outputSize, INTER_LINEAR);
		putText(legends[currentIndex * noOfImagePerRow + 5], "[6]: [5] - [3]", Point(1, 12), 1, .7, Scalar(255, 255, 255), 1);

		// remove very small contours which are noises
		Mat cleanSatAndFilled;
		removeSmallContours(cutWithEdge, cleanSatAndFilled, .005);
		resize(cleanSatAndFilled, win[currentIndex * noOfImagePerRow + 6], outputSize, INTER_LINEAR);
		putText(legends[currentIndex * noOfImagePerRow + 6], "[7]: Remove small contours", Point(1, 12), 1, .7, Scalar(255, 255, 255), 1);

		// find largest contour
		Mat finalLargestContourMaskNoFill, finalLargestContourMaskFilled;
		findLargestContour(cleanSatAndFilled, finalLargestContourMaskNoFill, finalLargestContourMaskFilled);
		resize(finalLargestContourMaskFilled, win[currentIndex * noOfImagePerRow + 7], outputSize, INTER_LINEAR);
		putText(legends[currentIndex * noOfImagePerRow + 7], "[8]: Filled longest contour in [7]", Point(1, 12), 1, .7, Scalar(255, 255, 255), 1);

		// draw bounding box
		Mat finalMask;
		Rect rect;
		images[i].copyTo(finalMask);
		boundingBox(finalLargestContourMaskFilled, finalMask, rect);
		resize(finalMask, win[currentIndex * noOfImagePerRow + 8], outputSize, INTER_LINEAR);
		putText(legends[currentIndex * noOfImagePerRow + 8], "[9]: Bounding box", Point(1, 12), 1, .7, Scalar(255, 255, 255), 1);

		// crop the image for recognition
		Mat toRecog = images[i](rect);
		resize(toRecog, win[currentIndex * noOfImagePerRow + 9], outputSize, INTER_LINEAR);
		putText(legends[currentIndex * noOfImagePerRow + 9], "[10]: For recognition", Point(1, 12), 1, .7, Scalar(255, 255, 255), 1);

		// show segmentation accuracy
		int* coordinate = coordinates[i];
		double area1 = rect.area();
		double area2 = (coordinate[2] - coordinate[0]) * (coordinate[3] - coordinate[1]);
		double x_diff = max({ 0, min({coordinate[2], rect.x + rect.width}) }) - max({ coordinate[0], rect.x });
		double y_diff = max({ 0, min({coordinate[3], rect.y + rect.height}) }) - max({coordinate[1], rect.y});
		double overlap = x_diff * y_diff;
		double area = area1 + area2 - overlap;
		double accurracy = overlap / area;
		total_accuracy += accurracy;
		putText(win[currentIndex * noOfImagePerRow + 10], "Accuracy: " + to_string(accurracy), Point(1, 12), 1, .7, Scalar(255, 255, 255), 1);
		putText(legends[currentIndex * noOfImagePerRow + 10], "[11]: Segmentation result", Point(1, 12), 1, .7, Scalar(255, 255, 255), 1);

		 //recognise the flag
		Mat feature;
		FeatureExtraction(toRecog, feature);
		subtract(feature, mean, feature);
		divide(feature, stdDev, feature);
		String actualLabel = fileNames[i].substr(0, fileNames[i].size() - 4);
		int actualLabelIndex = 0;
		for (int j = 0; j < classes.size(); j++) {
			if (actualLabel == classes[j]) {
				actualLabelIndex = j;
			}
		}
		int predictedLabelIndex = svm->predict(feature);
		String predictedLabel = classes[predictedLabelIndex];
		if (actualLabelIndex == predictedLabelIndex) {
			correctPrediction++;
		}
		cout << "Actual: " << actualLabelIndex << "\tPredicted: " << predictedLabelIndex << "\t(" << actualLabel << " as " << predictedLabel << ")" << endl;

		if (currentIndex == 99 || i == images.size() - 1) {
			imwrite(setName + " results " + to_string(totalResultImageCount++) + ".png", results(Range(0, outputSize.height * (currentIndex + 1) + SEPV * (currentIndex + 1)), Range::all()));
		}
	}
	cout << "Recognition Accuracy: " << correctPrediction / (double)images.size() << endl;
	cout << "Segmentation Accuracy: " << total_accuracy / (double)images.size() << endl;
	cout << endl << endl;
}

void avgSaturation(Mat source, Mat& result) {
	// blur the image
	Mat blur;
	GaussianBlur(source, blur, Size(3, 3), 0);

	// convert to hsv colour space
	Mat hsv;
	cvtColor(blur, hsv, COLOR_BGR2HSV);

	// split into H, S and V channels
	vector<Mat> channels;
	split(hsv, channels);

	// find the mean saturation value
	Scalar avg_saturation = mean(channels[1]);

	// Ssegment out colours with high (larger than mean) saturation only 
	Scalar saturationStart = Scalar(0, avg_saturation[0], 0), saturationEnd = Scalar(180, 255, 255);
	inRange(hsv, saturationStart, saturationEnd, result);

	// Segment out black and white colours
	Scalar blackHSVStart = Scalar(0, 0, 0), blackHSVEnd = Scalar(180, 130, 50);
	Scalar whiteHSVStart = Scalar(0, 0, 130), whiteHSVEnd = Scalar(180, 55, 255);

	// thresholding
	Mat blackMask, whiteMask;
	inRange(hsv, blackHSVStart, blackHSVEnd, blackMask);
	inRange(hsv, whiteHSVStart, whiteHSVEnd, whiteMask);

	// adding into the result
	result = result | blackMask | whiteMask;

	// Morph close to fill small lines in the flag
	morphologyEx(result, result, MORPH_CLOSE, kernel);

	// store the result
	cvtColor(result, result, COLOR_GRAY2BGR);
}

void cannySeg(Mat source, Mat& result) {
	// blur the image
	Mat blur;
	GaussianBlur(source, blur, Size(3, 3), 0);

	// convert to hsv colour space
	Mat hsv;
	cvtColor(blur, hsv, COLOR_BGR2HSV);

	// split into H, S, and V channels
	vector<Mat> channels;
	split(hsv, channels);

	// Apply Canny edge detection to the hue and saturation channel
	Mat edges, result1, result2;
	Canny(channels[0], result1, 1, 200);
	Canny(channels[1], result2, 1, 200);

	// combine both masks
	result = result1 | result2;

	// store the result
	cvtColor(result, result, COLOR_GRAY2BGR);
}

void findLargestContour(Mat source, Mat& noFillResult, Mat& filledResult) {
	// convert to single channel
	Mat gray;
	cvtColor(source, gray, COLOR_BGR2GRAY);

	// find contours
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(gray, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	// get largest contour
	double largestArea = 0;
	int largestContourIndex = -1;
	for (int i = 0; i < contours.size(); i++) {
		double area = contourArea(contours[i]);
		if (area > largestArea) {
			largestArea = area;
			largestContourIndex = i;
		}
	}

	// draw the result
	noFillResult = Mat::zeros(gray.size(), CV_8UC3);
	filledResult = Mat::zeros(gray.size(), CV_8UC3);
	drawContours(noFillResult, contours, largestContourIndex, Scalar(255, 255, 255), 1);
	drawContours(filledResult, contours, largestContourIndex, Scalar(255, 255, 255), FILLED);
}

void removeSmallContours(Mat source, Mat &result, double threshold) {
	// convert to single channel
	Mat gray;
	cvtColor(source, gray, COLOR_BGR2GRAY);

	// erode to remove small lines connecting the noises
	erode(gray, gray, kernel);

	// find contours
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(gray, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	// keep the contours with size larger than the threshold only
	double threshSize = threshold * gray.rows * gray.cols;
	result = Mat::zeros(gray.size(), CV_8UC3);
	for (int i = 0; i < contours.size(); i++) {
		double area = contourArea(contours[i]);
		if (area > threshSize) {
			drawContours(result, contours, i, Scalar(255, 255, 255), FILLED);
		}
	}

	// dilate to connect back the large contours
	dilate(result, result, kernel, Point(-1, -1), 3);
}

void boundingBox(Mat source, Mat& result, Rect& rect) {
	// convert to single channel
	Mat gray;
	cvtColor(source, gray, COLOR_BGR2GRAY);

	// find contours
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(gray, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	// get the bounding box of largest contour if exist
	if (contours.size() > 0) {
		rect = boundingRect(contours[0]);

		// draw on result
		rectangle(result, rect, Scalar(0, 255, 0), 3);
	}
}

void colourSegHSV(Mat source, Mat& result) {
	Mat blur, hsv, red1mask, red2mask, yellowmask, greenmask, bluemask, whitemask, blackmask, mask, segmentBGR;
	Mat kernel = (Mat_<unsigned char>(3, 3) 
		<< 1, 1, 1,
		1, 1, 1,
		1, 1, 1);

	// define colour ranges in HSV
	Scalar redStart1 = Scalar(0, 110, 140), redEnd1 = Scalar(15, 255, 255);
	Scalar redStart2 = Scalar(165, 110, 140), redEnd2 = Scalar(180, 255, 255);
	Scalar yellowStart1 = Scalar(15, 85, 50), yellowEnd1 = Scalar(40, 255, 255);
	Scalar greenStart1 = Scalar(50, 100, 100), greenEnd1 = Scalar(85, 255, 255);
	Scalar blueStart1 = Scalar(75, 70, 50), blueEnd1 = Scalar(140, 255, 255);
	Scalar whiteStart1 = Scalar(0, 0, 190), whiteEnd1 = Scalar(180, 80, 255);
	Scalar blackStart1 = Scalar(0, 0, 0), blackEnd1 = Scalar(180, 255, 45);

	// blur source image using gaussian blur
	medianBlur(source, blur, 3);

	// convert source image into HSV colour space
	cvtColor(blur, hsv, COLOR_BGR2HSV);

	// thresholding using the HSV colour ranges
	inRange(hsv, redStart1, redEnd1, red1mask);
	inRange(hsv, redStart2, redEnd2, red2mask);
	inRange(hsv, yellowStart1, yellowEnd1, yellowmask);
	inRange(hsv, greenStart1, greenEnd1, greenmask);
	inRange(hsv, blueStart1, blueEnd1, bluemask);
	inRange(hsv, whiteStart1, whiteEnd1, whitemask);
	inRange(hsv, blackStart1, blackEnd1, blackmask);
	mask = red1mask | red2mask | yellowmask | greenmask | bluemask | whitemask | blackmask;

	// use morphological close operation to fill small gaps between segmented regions
	dilate(mask, mask, kernel, Point(-1, -1), 3);
	morphologyEx(mask, mask, MORPH_CLOSE, kernel);

	// store result
	cvtColor(mask, result, COLOR_GRAY2BGR);
}

void FeatureExtraction(Mat image, Mat& features) {
	// Preprocessing
	Size fixedSize = Size(256, 256);
	resize(image, image, fixedSize, INTER_LINEAR);
	GaussianBlur(image, image, Size(3, 3), 0);
	Mat hsv;
	cvtColor(image, hsv, COLOR_BGR2HSV);

	// colour feature Extraction
	int numOfBins1 = 10;
	int minSat = 50;
	int minValue = 50;
	int numOfBins2 = 5;
	int numOfSepsHorizontal = 5;
	int numOfSepsVertical = 9;
	double middleAreaVertical = 0.3;
	double middleAreaHorizontal = 0.3;

	// Count the number of pixels in each bin in each separation according to their Hue
	int height = image.rows;
	int width = image.cols;
	int binSize1 = 180 / numOfBins1;
	int sepSizeHorizontal = height / numOfSepsHorizontal;
	int sepSizeVertical = width / numOfSepsVertical;
	int totalSepsVertical = numOfSepsVertical / 2 + numOfSepsVertical % 2;
	features = Mat(1, (numOfBins1 + numOfBins2) * (numOfSepsHorizontal + totalSepsVertical + 2), CV_32F);

	for (int i = 0; i < numOfBins1; i++) {
		int startThresh = i * binSize1;
		int endThresh = startThresh + binSize1 - 1;
		if (endThresh + binSize1 > 180) {
			endThresh = 180;
		}
		Mat mask;
		Scalar rangeStart(startThresh, minSat, minValue), rangeEnd(endThresh, 255, 255);
		for (int j = 0; j < numOfSepsHorizontal; j++) {
			int rowRangeStart = j * sepSizeHorizontal;
			int rowRangeEnd = rowRangeStart + sepSizeHorizontal;
			if (rowRangeEnd + sepSizeHorizontal > height) {
				rowRangeEnd = height;
			}
			inRange(hsv.rowRange(rowRangeStart, rowRangeEnd), rangeStart, rangeEnd, mask);
			features.at<float>((numOfBins1 + numOfBins2) * j + i) = countNonZero(mask) / (float)(width * (rowRangeEnd - rowRangeStart));
		}

		Mat mask1, mask2;
		for (int j = 0; j < totalSepsVertical; j++) {
			int colRangeStart1 = j * sepSizeVertical;
			int colRangeEnd1 = colRangeStart1 + sepSizeVertical;
			int colRangeStart2 = (numOfSepsVertical - j - 1) * sepSizeVertical;
			int colRangeEnd2 = colRangeStart2 + sepSizeVertical;
			inRange(hsv.colRange(colRangeStart1, colRangeEnd1), rangeStart, rangeEnd, mask1);
			inRange(hsv.colRange(colRangeStart2, colRangeEnd2), rangeStart, rangeEnd, mask2);
			features.at<float>((numOfBins1 + numOfBins2) * (numOfSepsHorizontal + j) + i) = (countNonZero(mask1) + countNonZero(mask2)) / (float)(height * (colRangeEnd1 - colRangeStart1) * 2);
		}

		int colRangeStart = width / 2 - width * middleAreaVertical / 2;
		int colRangeEnd = width / 2 + width * middleAreaVertical / 2;
		inRange(hsv.colRange(colRangeStart, colRangeEnd), rangeStart, rangeEnd, mask);
		features.at<float>((numOfBins1 + numOfBins2) * (numOfSepsHorizontal + totalSepsVertical) + i) = countNonZero(mask) / (float)(height * (colRangeEnd - colRangeStart));

		int rowRangeStart = height / 2 - height * middleAreaHorizontal / 2;
		int rowRangeEnd = height / 2 + height * middleAreaHorizontal / 2;
		inRange(hsv.rowRange(rowRangeStart, rowRangeEnd), rangeStart, rangeEnd, mask);
		features.at<float>((numOfBins1 + numOfBins2) * (numOfSepsHorizontal + totalSepsVertical + 1) + i) = countNonZero(mask) / (float)(width * (rowRangeEnd - rowRangeStart));
	}

	// for black and white according to their value
	int binSize2 = 255 / numOfBins2;
	for (int i = 0; i < numOfBins2; i++) {
		int startThresh = i * binSize2;
		int endThresh = startThresh + binSize2 - 1;
		if (endThresh + binSize2 > 255) {
			endThresh = 255;
		}
		Mat mask;
		Scalar rangeStart(0, 0, startThresh), rangeEnd(180, minSat - 1, endThresh);
		for (int j = 0; j < numOfSepsHorizontal; j++) {
			int rowRangeStart = j * sepSizeHorizontal;
			int rowRangeEnd = rowRangeStart + sepSizeHorizontal;
			if (rowRangeEnd + sepSizeHorizontal > height) {
				rowRangeEnd = height;
			}
			inRange(hsv.rowRange(rowRangeStart, rowRangeEnd), startThresh, endThresh, mask);
			features.at<float>((numOfBins1 + numOfBins2) * j + numOfBins1 + i) = countNonZero(mask) / (float)(width * (rowRangeEnd - rowRangeStart));
		}

		Mat mask1, mask2;
		for (int j = 0; j < totalSepsVertical; j++) {
			int colRangeStart1 = j * sepSizeVertical;
			int colRangeEnd1 = colRangeStart1 + sepSizeVertical;
			int colRangeStart2 = (numOfSepsVertical - j - 1) * sepSizeVertical;
			int colRangeEnd2 = colRangeStart2 + sepSizeVertical;
			inRange(hsv.colRange(colRangeStart1, colRangeEnd1), rangeStart, rangeEnd, mask1);
			inRange(hsv.colRange(colRangeStart2, colRangeEnd2), rangeStart, rangeEnd, mask2);
			features.at<float>((numOfBins1 + numOfBins2) * (numOfSepsHorizontal + j) + numOfBins1 + i) = (countNonZero(mask1) + countNonZero(mask2)) / (float)(height * (colRangeEnd1 - colRangeStart1) * 2);
		}

		int colRangeStart = width / 2 - width * middleAreaVertical / 2;
		int colRangeEnd = width / 2 + width * middleAreaVertical / 2;
		inRange(hsv.colRange(colRangeStart, colRangeEnd), rangeStart, rangeEnd, mask);
		features.at<float>((numOfBins1 + numOfBins2) * (numOfSepsHorizontal + totalSepsVertical) + numOfBins1 + i) = countNonZero(mask) / (float)(height * (colRangeEnd - colRangeStart));

		int rowRangeStart = height / 2 - height * middleAreaHorizontal / 2;
		int rowRangeEnd = height / 2 + height * middleAreaHorizontal / 2;
		inRange(hsv.rowRange(rowRangeStart, rowRangeEnd), rangeStart, rangeEnd, mask);
		features.at<float>((numOfBins1 + numOfBins2) * (numOfSepsHorizontal + totalSepsVertical + 1) + numOfBins1 + i) = countNonZero(mask) / (float)(width * (rowRangeEnd - rowRangeStart));
	}
}

void createWindowPartition(Mat srcI, Mat& largeWin, Mat win[], Mat legends[], int noOfImagePerCol,
	int noOfImagePerRow, int sepH, int sepV) {
	// 1st input: source input image
	// 2nd: the created larger window
	// 3th: means to access each sub window
	// 4th: means to access each legend window
	// 5rd, 6th: Obvious
	// 7th: separating space between 2 images in horizontal direction
	// 8th: separating space between 2 images in vertial direction

	int		rows = srcI.rows, cols = srcI.cols, winI = 0, winsrcI = 0;
	Size	sRXC((cols + sepH) * noOfImagePerRow - sepH, (rows + sepV) * noOfImagePerCol),
		s(cols, sepV);

	largeWin = Mat::ones(sRXC, srcI.type()) * 64;
	for (int i = 0; i < noOfImagePerCol; i++)
		for (int j = 0; j < noOfImagePerRow; j++)
			win[winI++] = largeWin(Range((rows + sepV) * i, (rows + sepV) * i + rows),
				Range((cols + sepH) * j, (cols + sepH) * j + cols));

	for (int bg = 20, i = 0; i < noOfImagePerCol; i++)
		for (int j = 0; j < noOfImagePerRow; j++) {
			legends[winsrcI] = largeWin(Range((rows + sepV) * i + rows, (rows + sepV) * (i + 1)),
				Range((cols + sepH) * j, (cols + sepH) * j + cols));
			legends[winsrcI] = Scalar(bg, bg, bg);
			bg += 30;
			if (bg > 80) bg = 20;
			winsrcI++;
		}
}