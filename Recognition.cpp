#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

using namespace std;
using namespace cv;
using namespace cv::ml;

void featureExtraction(Mat image, Mat& features);

int main() {
	string modelName = "all rows";

	// train:test:val ratio
	float valRatio = 0;
	float testRatio = 0;
	int minTreshold = 100; // minimum number of images required to give validation and test set at least one image

	// read images and labels
	vector<String> classes = {};
	vector<int> classesImagesCount = {};
	vector<Mat> images = {};
	vector<int> labels = {};
	{
		cout << "Loading image..." << endl;
		system("dir \"Inputs\\cropped recognition dataset\\\" /b > dirList.txt");
		ifstream readClasses("dirList.txt");
		string className;
		while (getline(readClasses, className)) {
			classes.push_back(className);
			string command = "dir \"Inputs\\cropped recognition dataset\\" + className + "\\*.png\" /b > dirList2.txt";
			system(command.c_str());

			ifstream readImages("dirList2.txt");
			string imageName;
			int label = classes.size() - 1;
			int count = 0;
			while (getline(readImages, imageName)) {
				Mat image = imread("Inputs\\cropped recognition dataset\\" + className + "\\" + imageName);
				images.push_back(image);
				labels.push_back(label);
				count++;
			}
			classesImagesCount.push_back(count);
			readImages.close();
		}
		readClasses.close();
		system("del dirList.txt");
		system("del dirList2.txt");
	}


	// split train, validation, and test sets according to the ratio
	vector<Mat> trainImages = {};
	vector<int> trainImagesLabels = {};
	vector<Mat> valImages = {};
	vector<int> valImagesLabels = {};
	vector<Mat> testImages = {};
	vector<int> testImagesLabels = {};
	{
		cout << "Splitting train/ val/ test sets..." << endl;
		srand(42);
		int allImagesStartIndex = 0;
		for (int i = 0; i < classes.size(); i++) {
			bool* assigned = new bool[classesImagesCount[i]]{ false };
			int currentClassValImages = classesImagesCount[i] * valRatio;
			if (currentClassValImages == 0 && classesImagesCount[i] > minTreshold) {
				currentClassValImages++;
			}
			for (int j = 0; j < currentClassValImages; j++) {
				int randint = rand() % classesImagesCount[i];
				while (assigned[randint]) {
					randint++;
					if (randint >= classesImagesCount[i]) {
						randint = 0;
					}
				}
				assigned[randint] = true;
				valImages.push_back(images[allImagesStartIndex + randint]);
				valImagesLabels.push_back(labels[allImagesStartIndex + randint]);
			}

			int currentClassTestImages = classesImagesCount[i] * testRatio;
			if (currentClassTestImages == 0 && classesImagesCount[i] > minTreshold) {
				currentClassTestImages++;
			}
			for (int j = 0; j < currentClassTestImages; j++) {
				int randint = rand() % classesImagesCount[i];
				while (assigned[randint]) {
					randint++;
					if (randint >= classesImagesCount[i]) {
						randint = 0;
					}
				}
				assigned[randint] = true;
				testImages.push_back(images[allImagesStartIndex + randint]);
				testImagesLabels.push_back(labels[allImagesStartIndex + randint]);
			}

			for (int j = 0; j < classesImagesCount[i]; j++) {
				if (!assigned[j]) {
					trainImages.push_back(images[allImagesStartIndex + j]);
					trainImagesLabels.push_back(labels[allImagesStartIndex + j]);
				}
			}
			allImagesStartIndex += classesImagesCount[i];
		}
		vector<int> shuffled = {};
		for (int i = 0; i < trainImages.size(); i++) {
			shuffled.push_back(i);
		}
		mt19937 rng(42);
		shuffle(shuffled.begin(), shuffled.end(), rng);
		vector<Mat> shuffledImages = {};
		vector<int> shuffledLabels = {};
		for (int i = 0; i < shuffled.size(); i++) {
			shuffledImages.push_back(trainImages[shuffled[i]]);
			shuffledLabels.push_back(trainImagesLabels[shuffled[i]]);
		}
		trainImages = shuffledImages;
		trainImagesLabels = shuffledLabels;
	}


	// feature extraction
	Mat trainFeatures;
	Mat trainLabels;
	Mat valFeatures;
	Mat valLabels;
	Mat testFeatures;
	Mat testLabels;
	{
		cout << "Extracting features..." << endl;
		for (int i = 0; i < trainImages.size(); i++) {
			Mat feature;
			featureExtraction(trainImages[i], feature);
			trainFeatures.push_back(feature);
			trainLabels.push_back(trainImagesLabels[i]);
		}

		for (int i = 0; i < valImages.size(); i++) {
			Mat feature;
			featureExtraction(valImages[i], feature);
			valFeatures.push_back(feature);
			valLabels.push_back(valImagesLabels[i]);
		}

		for (int i = 0; i < testImages.size(); i++) {
			Mat feature;
			featureExtraction(testImages[i], feature);
			testFeatures.push_back(feature);
			testLabels.push_back(testImagesLabels[i]);
		}
	}


	// normalisation
	Scalar mean, stdDev;
	meanStdDev(trainFeatures, mean, stdDev);
	subtract(trainFeatures, mean, trainFeatures);
	divide(trainFeatures, stdDev, trainFeatures);
	if (valLabels.total() != 0) {
		subtract(valFeatures, mean, valFeatures);
		divide(valFeatures, stdDev, valFeatures);
	}
	if (testLabels.total() != 0) {
		subtract(testFeatures, mean, testFeatures);
		divide(testFeatures, stdDev, testFeatures);
	}


	// train and test the SVM classifier
	cout << "Training classifier..." << endl;
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::LINEAR);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
	svm->train(trainFeatures, ROW_SAMPLE, trainLabels);


	// save model and normalisation details
	svm->save("Trained models\\SVM - " + modelName + ".xml");
	ofstream file("Trained models\\Norm - " + modelName + ".txt");
	file << mean[0] << endl;
	file << stdDev[0] << endl;
	file.close();
	ofstream labelsFile("Trained models\\Labels - " + modelName + ".txt");
	for (int i = 0; i < classes.size(); i++) {
		labelsFile << classes[i] << endl;
	}
	labelsFile.close();

	// Predict using the SVM classifier
	cout << "Training set:" << endl;
	int trainCorrectCount = 0;
	Mat predictedTrainLabels;
	svm->predict(trainFeatures, predictedTrainLabels);
	for (int i = 0; i < trainLabels.total(); i++) {
		cout << "Actual: " << trainLabels.at<int>(i) << "\tPredicted: " << predictedTrainLabels.at<float>(i) << "\t(" << classes[trainLabels.at<int>(i)] << " as " << classes[predictedTrainLabels.at<float>(i)] << ")" << endl;
		if (predictedTrainLabels.at<float>(i) == trainLabels.at<int>(i)) {
			trainCorrectCount++;
		}
	}
	cout << "training accuracy: " << trainCorrectCount / (double)trainLabels.total() << endl;
	cout << endl << endl;

	if (valLabels.total() != 0) {
		cout << "Validation set:" << endl;
		int valCorrectCount = 0;
		Mat predictedValLabels;
		svm->predict(valFeatures, predictedValLabels);
		for (int i = 0; i < valLabels.total(); i++) {
			cout << "Actual: " << valLabels.at<int>(i) << "\tPredicted: " << predictedValLabels.at<float>(i) << "\t(" << classes[valLabels.at<int>(i)] << " as " << classes[predictedValLabels.at<float>(i)] << ")" << endl;
			if (predictedValLabels.at<float>(i) == valLabels.at<int>(i)) {
				valCorrectCount++;
			}
		}
		cout << "validation accuracy: " << valCorrectCount / (double)valLabels.total() << endl;
		cout << endl << endl;
	}

	if (testLabels.total() != 0) {
		cout << "Test Set:" << endl;
		int testCorrectCount = 0;
		Mat predictedTestLabels;
		svm->predict(testFeatures, predictedTestLabels);
		for (int i = 0; i < testLabels.total(); i++) {
			cout << "Actual: " << testLabels.at<int>(i) << "\tPredicted: " << predictedTestLabels.at<float>(i) << "\t(" << classes[testLabels.at<int>(i)] << " as " << classes[predictedTestLabels.at<float>(i)] << ")" << endl;
			if (predictedTestLabels.at<float>(i) == testLabels.at<int>(i)) {
				testCorrectCount++;
			}
		}
		cout << "testing accuracy: " << testCorrectCount / (double)testLabels.total() << endl;
		cout << endl << endl;
	}

	return 0;
}

void featureExtraction(Mat image, Mat &features) {
	// Preprocessing
	Size fixedSize = Size(256, 256);
	resize(image, image, fixedSize, INTER_LINEAR);
	GaussianBlur(image, image, Size(3,3), 0);
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
			features.at<float>((numOfBins1 + numOfBins2) * j + numOfBins1 + i) = countNonZero(mask) / (float)(width*(rowRangeEnd - rowRangeStart));
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
