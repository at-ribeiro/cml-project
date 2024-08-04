#include "ofApp.h"
#include "ofxCvHaarFinder.h"
#include "ofxCv.h"

using namespace cv;
using namespace ofxCv;

//--------------------------------------------------------------
void ofApp::setup() {

	ofResetElapsedTimeCounter();

	ofBackground(25, 19, 28);

	currentWindow = STANDBY;
	cameraToggle = false;
	fullImageToggle = false;
	fullVideoToggle = false;
	workspaceMenuToggle = false;
	plusImageToggle = false;
	plusVideoToggle = false;
	trashToggle = false;

	peopleNum = 0;

	//load the yolo5n model and the classes file which has the items it is trained to detect listed
	classify.setup("yolov5n.onnx", "classes.txt", false);

	//open the webcam
	grabber.setup(1000, 600);

	//icons
	plusIcon.load("ui/plus.png");
	plusIcon.resize(64, 64);
	deleteIcon.load("ui/trash.png");
	deleteIcon.resize(64, 64);
	plusMenu.load("ui/plusMenu.png");
	plusMenu.resize(90, 250);
	imageIcon.load("ui/image.png");
	imageIcon.resize(64, 64);
	videoIcon.load("ui/video.png");
	videoIcon.resize(64, 64);
	facialDetectionIcon.load("ui/facial_detection.png");
	facialDetectionIcon.resize(64, 64);
	homeIcon.load("ui/home.png");
	homeIcon.resize(64, 64);
	standByImage.load("ui/standBy.jpg");
	personIcon.load("ui/person.png");
	personIcon.resize(32, 32);
	previousIcon.load("ui/previous.png");
	previousIcon.resize(64, 64);
	nextIcon.load("ui/next.png");
	nextIcon.resize(64, 64);
	playIcon.load("ui/play.png");
	playIcon.resize(64, 64);
	pauseIcon.load("ui/pause.png");
	pauseIcon.resize(64, 64);
	arrowUpIcon.load("ui/arrow_up.png");
	arrowUpIcon.resize(64, 64);
	arrowDownIcon.load("ui/arrow_down.png");
	arrowDownIcon.resize(64, 64);
	square_plus.load("ui/square_plus.png");
	square_plus.resize(50, 50);
	grid.load("ui/grid.png");
	grid.resize(300, 195);

	//fonts
	font.load("fonts/arial.ttf", 20);

	//image dir
	currentImage = 0;
	imageDir.listDir("images/");
	imageDir.allowExt("jpg");
	imageDir.sort();

	if (imageDir.size()) {
		images.assign(imageDir.size(), ofImage());
	}

	for (int i = 0; i < (int)imageDir.size(); i++) {
		images[i].load(imageDir.getPath(i));
	}
	
	//video dir
	isPlaying = false;
	videoState = false;
	currentVideo = 0;
	videoDir.listDir("videos");
	videoDir.allowExt("mp4");
	videoDir.sort(); 

	if (videoDir.size()) {
		videos.assign(videoDir.size(), ofVideoPlayer());
	}

	ofDirectory xmlDir("imgXmls");

	if (!xmlDir.exists()) {
		std::cout << "imgXmls directory non existent " + xmlDir.getAbsolutePath() << std::endl;
	}

	xmlDir.allowExt("xml");
	xmlDir.listDir();

	for (int i = 0; i < (int)imageDir.size(); i++) {
		images[i].load(imageDir.getPath(i));
		ofxXmlSettings xmlnew;
		if (xmlDir.size() <= i || !xmlnew.loadFile(xmlDir.getPath(i))) {
			std::cout << "XML did not exist or wrong format, creating...." << std::endl;
			handleXMLDoesNotExist(images[i],i);
		}
		else {
			std::cout << "XML " + xmlDir.getPath(i) + " successfully read" << std::endl;
		}
	}

	ofDirectory xmlVidDir("vidXmls");

	if (!xmlVidDir.exists()) {
		std::cout << "vidXmls directory non existent " + xmlVidDir.getAbsolutePath() << std::endl;
	}

	xmlVidDir.allowExt("xml");
	xmlVidDir.listDir();

	for (int i = 0; i < (int)videoDir.size(); i++) {
		videos[i].load(videoDir.getPath(i));
		videos[i].setLoopState(OF_LOOP_NORMAL);
		ofxXmlSettings xmlnew;
		if (xmlVidDir.size() <= i || !xmlnew.loadFile(xmlVidDir.getPath(i))) {
			std::cout << "XML did not exist or wrong format, creating...." << std::endl;
			handleXMLDoesNotExist(videos[i], i);
		}
		else {
			std::cout << "XML " + xmlVidDir.getPath(i) + " successfully read" << std::endl;
		}
	}

	reorderImgLib();

}

void ofApp::handleXMLDoesNotExist(ofImage image, int i) {

	if(!image.isAllocated())
		return;

	XML.addTag("root");
	XML.pushTag("root");

	//tags---------------------------------------------------------------------------------------------------------
	XML.addTag("tags");
	XML.pushTag("tags");
	XML.addValue("tag1", "exampleTag(to add more, add more tags with tag2, tag3,..., tagN)");
	XML.popTag();

	//luminance-----------------------------------------------------------------------------------------------------
	ofxCvColorImage colorImg;
	colorImg.setFromPixels(image.getPixels());
	Mat mat = toCv(colorImg.getPixels());

	Mat grayMat;
	cvtColor(mat, grayMat, cv::COLOR_RGB2GRAY);
	double averageLuminance = cv::mean(grayMat)[0];
	XML.addValue("luminance", averageLuminance);

	//color-----------------------------------------------------------------------------------------------------
	XML.addTag("color");
	XML.pushTag("color");

	vector<Mat> channels;
	cv::split(mat, channels);

	Mat red = channels[0];
	Mat green = channels[1];
	Mat blue = channels[2];

	double averageRed = cv::mean(red)[0];
	double averageGreen = cv::mean(green)[0];
	double averageBlue = cv::mean(blue)[0];

	XML.addValue("red", averageRed);
	XML.addValue("green", averageGreen);
	XML.addValue("blue", averageBlue);

	XML.popTag(); //color
	

	//Edge Distribution-----------------------------------------------------------------------------------------------------
	XML.addTag("Edge_Distribution");
	XML.pushTag("Edge_Distribution");
	
	cv::Mat ver_edg_fil = (cv::Mat_<double>(2, 2) << 1, -1, 1, -1);
	cv::Mat hor_edg_fil = (cv::Mat_<double>(2, 2) << 1, 1, -1, -1);
	cv::Mat dia45_edg_fil = (cv::Mat_<double>(2, 2) << sqrt(2), 0, 0, -sqrt(2));
	cv::Mat dia135_edg_fil = (cv::Mat_<double>(2, 2) << 0, sqrt(2), -sqrt(2), 0);
	cv::Mat nond_edg_fil = (cv::Mat_<double>(2, 2) << 2, -2, -2, 2);

	cv::Mat ver_edg_img, hor_edg_img, dia45_edg_img, dia135_edg_img, nond_edg_img;

	cv::filter2D(mat, ver_edg_img, -1, ver_edg_fil);
	cv::filter2D(mat, hor_edg_img, -1, hor_edg_fil);
	cv::filter2D(mat, dia45_edg_img, -1, dia45_edg_fil);
	cv::filter2D(mat, dia135_edg_img, -1, dia135_edg_fil);
	cv::filter2D(mat, nond_edg_img, -1, nond_edg_fil);

	XML.addValue("Vertical_Edge", cv::mean(ver_edg_img)[0]);
	XML.addValue("Horizontal_Edge", cv::mean(hor_edg_img)[0]);
	XML.addValue("Edge_45_Degree", cv::mean(dia45_edg_img)[0]);
	XML.addValue("Edge_135_Degree", cv::mean(dia135_edg_img)[0]);
	XML.addValue("Non-directional_Edge", cv::mean(nond_edg_img)[0]);

	XML.popTag(); //Edge Distribution



	//texture-----------------------------------------------------------------------------------------------------
	XML.addTag("texture");
	XML.pushTag("texture");

	int nOrientations = 6;
	int nFrequencies = 4;

	vector<int> orientations = { 0, 30, 60, 90, 120, 150 };
	vector<int> frequencies = { 1, 25, 50, 100};

	Size ksize(21, 21);
	double sigma = 5.0;
	double gamma = 0.5;

	for (int i = 0; i < nOrientations; i++) {
		
		double theta = orientations[i];

		for (int j = 0; j < nFrequencies; j++) {
			double lambda = frequencies[j];
			Mat kernel = getGaborKernel(ksize, sigma, theta, lambda, gamma);
			Mat gaborImg;
			filter2D(mat, gaborImg, -1, kernel);

			XML.addValue("Orientation-" + ofToString(theta) + "_Frequency-" + ofToString(lambda), cv::mean(gaborImg)[0]);
		}

	}

	XML.popTag(); //texture

	
	//objects && nFaces-----------------------------------------------------------------------------------------------------

	XML.addTag("numFaces");
	XML.pushTag("numFaces");

	vector<yolo5ImageClassify::Result> objects = classify.classifyFrame(mat);

	std::map<std::string, int> objectCount;

	for (auto obj : objects) {
		std::string objectType = obj.label;
		objectCount[objectType]++;
	}
		
	XML.addValue("faces", objectCount["person"]);

	XML.popTag(); //numFaces

	XML.addTag("objects");
	XML.pushTag("objects");

	for (auto obj : objectCount) {
		if (obj.second != 0) {//sometimes person 0 will show up
			string noSpace = obj.first;
			std::replace(noSpace.begin(), noSpace.end(), ' ', '_');
			XML.addValue(noSpace, obj.second);
		}
	}

	XML.popTag(); //objects

	XML.popTag(); //root

	XML.save("imgXmls/" + imageDir.getName(i) + ".xml");

	XML.clear();


}

void ofApp::handleXMLDoesNotExist(ofVideoPlayer video, int i) {
	if (!video.isLoaded())
		return;
	
	ofImage frame;
	video.firstFrame();

	// Wait for the video to load the frame
	for (int j = 0; j < 10; ++j) {
		video.update();
		if (video.isFrameNew()) {
			break;
		}
		ofSleepMillis(100); // Sleep to give the video some time to load
	}

	frame.setFromPixels(video.getPixels());

	if (!frame.isAllocated())
		return;

	XML.addTag("root");
	XML.pushTag("root");

	//tags---------------------------------------------------------------------------------------------------------
	XML.addTag("tags");
	XML.pushTag("tags");
	XML.popTag();

	//luminance-----------------------------------------------------------------------------------------------------
	ofxCvColorImage colorImg;
	colorImg.setFromPixels(frame.getPixels());
	Mat mat = toCv(colorImg.getPixels());

	Mat grayMat;
	cvtColor(mat, grayMat, cv::COLOR_RGB2GRAY);
	double averageLuminance = cv::mean(grayMat)[0];
	XML.addValue("luminance", averageLuminance);

	//color-----------------------------------------------------------------------------------------------------
	XML.addTag("color");
	XML.pushTag("color");

	vector<Mat> channels;
	cv::split(mat, channels);

	Mat red = channels[0];
	Mat green = channels[1];
	Mat blue = channels[2];

	double averageRed = cv::mean(red)[0];
	double averageGreen = cv::mean(green)[0];
	double averageBlue = cv::mean(blue)[0];

	XML.addValue("red", averageRed);
	XML.addValue("green", averageGreen);
	XML.addValue("blue", averageBlue);

	XML.popTag();

	//Edge Distribution-----------------------------------------------------------------------------------------------------
	XML.addTag("Edge_Distribution");
	XML.pushTag("Edge_Distribution");

	cv::Mat ver_edg_fil = (cv::Mat_<double>(2, 2) << 1, -1, 1, -1);
	cv::Mat hor_edg_fil = (cv::Mat_<double>(2, 2) << 1, 1, -1, -1);
	cv::Mat dia45_edg_fil = (cv::Mat_<double>(2, 2) << sqrt(2), 0, 0, -sqrt(2));
	cv::Mat dia135_edg_fil = (cv::Mat_<double>(2, 2) << 0, sqrt(2), -sqrt(2), 0);
	cv::Mat nond_edg_fil = (cv::Mat_<double>(2, 2) << 2, -2, -2, 2);

	cv::Mat ver_edg_img, hor_edg_img, dia45_edg_img, dia135_edg_img, nond_edg_img;

	cv::filter2D(mat, ver_edg_img, -1, ver_edg_fil);
	cv::filter2D(mat, hor_edg_img, -1, hor_edg_fil);
	cv::filter2D(mat, dia45_edg_img, -1, dia45_edg_fil);
	cv::filter2D(mat, dia135_edg_img, -1, dia135_edg_fil);
	cv::filter2D(mat, nond_edg_img, -1, nond_edg_fil);

	XML.addValue("Vertical_Edge", cv::mean(ver_edg_img)[0]);
	XML.addValue("Horizontal_Edge", cv::mean(hor_edg_img)[0]);
	XML.addValue("Edge_45_Degree", cv::mean(dia45_edg_img)[0]);
	XML.addValue("Edge_135_Degree", cv::mean(dia135_edg_img)[0]);
	XML.addValue("Non-directional_Edge", cv::mean(nond_edg_img)[0]);

	XML.popTag();


	//texture-----------------------------------------------------------------------------------------------------
	XML.addTag("texture");
	XML.pushTag("texture");

	int nOrientations = 6;
	int nFrequencies = 4;

	vector<int> orientations = { 0, 30, 60, 90, 120, 150 };
	vector<int> frequencies = { 1, 25, 50, 100 };

	Size ksize(21, 21);
	double sigma = 5.0;
	double gamma = 0.5;

	for (int i = 0; i < nOrientations; i++) {

		double theta = orientations[i];

		for (int j = 0; j < nFrequencies; j++) {
			double lambda = frequencies[j];
			Mat kernel = getGaborKernel(ksize, sigma, theta, lambda, gamma);
			Mat gaborImg;
			filter2D(mat, gaborImg, -1, kernel);

			XML.addValue("Orientation-" + ofToString(theta) + "_Frequency-" + ofToString(lambda), cv::mean(gaborImg)[0]);
		}

	}

	XML.popTag();

	//objects && nFaces-----------------------------------------------------------------------------------------------------

	XML.addTag("numFaces");
	XML.pushTag("numFaces");

	vector<yolo5ImageClassify::Result> objects = classify.classifyFrame(mat);

	std::map<std::string, int> objectCount;

	for (auto obj : objects) {
		std::string objectType = obj.label;
		objectCount[objectType]++;
	}

	XML.addValue("faces", objectCount["person"]);

	XML.popTag();

	XML.addTag("objects");
	XML.pushTag("objects");

	for (auto obj : objectCount) {
		if (obj.second != 0) {//sometimes person 0 will show up
			string noSpace = obj.first;
			std::replace(noSpace.begin(), noSpace.end(), ' ', '_');
			XML.addValue(noSpace, obj.second);
		}
	}

	XML.popTag();

	XML.popTag(); //root

	XML.save("vidXmls/" + videoDir.getName(i) + ".xml");

	XML.clear();

}

void ofApp::addXmlTags(int index, bool isImage) {
	XML.clear();
	if (isImage) {
		ofDirectory xmlImgDir("imgXmls");

		xmlImgDir.allowExt("xml");
		xmlImgDir.listDir();

		if (!XML.loadFile(xmlImgDir.getPath(index))) {
			std::cout << "XML does not exist" << std::endl;
		}
	}	
	else {
		ofDirectory xmlVidDir("vidXmls");

		xmlVidDir.allowExt("xml");
		xmlVidDir.listDir();

		if (!XML.load(xmlVidDir.getPath(index))) {
			std::cout << "XML does not exist" << std::endl;
		}
	}

	XML.pushTag("root");
	XML.pushTag("tags");

	vector<string> tags;
	bool cont = true;
	int i = 0;
	while (cont) {
		i++;
		string tagvalue = "tag" + std::to_string(i);
		string tag = XML.getValue(tagvalue, "");

		if (tag != "") {
			tags.push_back(tag);
		}
		else {
			cont = false;
		}
	}

	XML.popTag();
	XML.popTag();
	XML.clear();
	
	for (string tag : tags) {
		selectedTags.push_back(tag);
		std::cout << tag << std::endl;
	}
}

bool ofApp::tagInVector(string tag) {

	for (string t : selectedTags) {
		if (t == tag) {
			return true;
		}
	}

	return false;
}


//--------------------------------------------------------------
void ofApp::update(){
	
	uint64_t time = ofGetElapsedTimeMillis();
	
	
	if (time >= 5000 || currentWindow == STANDBY) {

		ofResetElapsedTimeCounter();

		//get the ofPixels and convert to an ofxCvColorImage
		auto pixels = grabber.getPixels();
		colorImg.setFromPixels(pixels);

		//get the ofCvColorImage as a cv::Mat image to pass to the classifier
		auto cvMat = cv::cvarrToMat(colorImg.getCvImage());

		//get the restuls as a vector of detected items.
		//each result has an ofRectangle for the bounds, a label which identifies the object and the confidence of the classifier
		results = classify.classifyFrame(cvMat);

		int aux = 0;

		for (auto res : results) {
			auto rect = res.rect;
			if (res.label == "person") {
				aux++;
			}

		}

		peopleNum = aux;

	}

	grabber.update();

	switch (currentWindow) {
		case 0: {
			
			if (peopleNum > 0) {
				currentWindow = WORKSPACE;
				isPlaying = false;
			}
			break;
		}
		case 2: {
	
			if (peopleNum == 0) {
				currentWindow = STANDBY;
				isPlaying = false;
			}
			break;
		}
		case 4 : {
			if (peopleNum == 0) {
				currentWindow = STANDBY;
				isPlaying = false;
			}
			if (videoDir.size() > 0) {
				videos[currentVideo].update();
			}
			break;
		}
		case 6: {
			if (peopleNum == 0) {
				currentWindow = STANDBY;
				isPlaying = false;
			}
			if (videoDir.size() > 0) {
				videos[currentVideo].update();
			}
			break;
		}
		default: {
			if (peopleNum == 0) {
				currentWindow = STANDBY;
				isPlaying = false;
			}
			break;
		}
	}
}



//--------------------------------------------------------------
void ofApp::draw(){
	switch (currentWindow) {
		case 0: {
			drawStandByScreen();
			break;
		}
		case 1: {
			drawWorkspace();
			break;
		}
		case 2: {
			drawFacialDetection();
			break;
		}
		case 3: {
			drawImageLibrary();
			break;
		}
		case 4: {
			drawVideoLibrary();
			break;
		}
		case 5: {
			drawFullImage();
			break;
		}
		case 6: {
			drawFullVideo();
			break;
		}
	}
	
}

void ofApp::drawFullVideo() {
	float width, height, widthRatio, heightRatio, ratio;
	ofVideoPlayer video;
	if (videoDir.size() > 0) {
		video = videos[currentVideo];

		widthRatio = ofGetWidth() / video.getWidth();
		heightRatio = ofGetHeight() / video.getHeight();

		if (widthRatio < heightRatio) {
			ratio = widthRatio;
		}
		else {
			ratio = heightRatio;
		}

		width = video.getWidth() * ratio;
		height = video.getHeight() * ratio;

		video.draw((ofGetWidth() - width) / 2, (ofGetHeight() - height) / 2, width, height);
	}

}

void ofApp::drawFullImage() {
	float width, height, widthRatio, heightRatio, ratio;
	ofImage image;
	if (imageDir.size() > 0) {
		image = imgLib[currentLibImage];

		widthRatio = ofGetWidth() / image.getWidth();
		heightRatio = ofGetHeight() / image.getHeight();

		if (widthRatio < heightRatio) {
			ratio = widthRatio;
		}
		else {
			ratio = heightRatio;
		}


		width = image.getWidth() * ratio;
		height = image.getHeight() * ratio;

		
		image.resize(width, height);
		image.draw((ofGetWidth() - width) / 2, (ofGetHeight() - height) / 2);
	}
}

void ofApp::drawVideoLibrary() {
	float videoHeight = ofGetHeight() * 0.5, videoWidth;
	ofVideoPlayer video;
	if (videoDir.size() > 0) {
		video = videos[currentVideo];
		videoWidth = video.getWidth() * videoHeight / video.getHeight();
		video.draw(ofGetWidth() * 0.35 - videoWidth/2, ofGetHeight() * 0.1, videoWidth, videoHeight);
	}

	if (isPlaying) {
		pauseIcon.draw(ofGetWidth() * 0.35 - 64/2, ofGetHeight() * 0.65);
	}
	else {
		playIcon.draw(ofGetWidth() * 0.35 - 64/2, ofGetHeight() * 0.65);
	}

	previousIcon.draw(ofGetWidth() * 0.35 - 64 * 2 - 20, ofGetHeight() * 0.65);
	nextIcon.draw(ofGetWidth() * 0.35 + 64 + 20, ofGetHeight() * 0.65);
	homeIcon.draw(ofGetWidth() * 0.05, ofGetHeight() * 0.85);
	plusIcon.draw(ofGetWidth() * 0.95 - 64, ofGetHeight() * 0.85);

	if (plusVideoToggle) {
		grid.draw(ofGetWidth() * 0.95 - grid.getWidth(), ofGetHeight() * 0.725 - grid.getHeight() / 2 - 64 / 1.7);

		square_plus.draw(ofGetWidth() * 0.95 - 64, ofGetHeight() * 0.725);
		square_plus.draw(ofGetWidth() * 0.95 - 64 * 2.15, ofGetHeight() * 0.725);
		square_plus.draw(ofGetWidth() * 0.95 - 64 * 3.3, ofGetHeight() * 0.725);
		square_plus.draw(ofGetWidth() * 0.95 - 64 * 4.45, ofGetHeight() * 0.725);

		square_plus.draw(ofGetWidth() * 0.95 - 64, ofGetHeight() * 0.725 - 64);
		square_plus.draw(ofGetWidth() * 0.95 - 64 * 2.15, ofGetHeight() * 0.725 - 64);
		square_plus.draw(ofGetWidth() * 0.95 - 64 * 3.3, ofGetHeight() * 0.725 - 64);
		square_plus.draw(ofGetWidth() * 0.95 - 64 * 4.45, ofGetHeight() * 0.725 - 64);

		square_plus.draw(ofGetWidth() * 0.95 - 64, ofGetHeight() * 0.725 - 64 * 2);
		square_plus.draw(ofGetWidth() * 0.95 - 64 * 2.15, ofGetHeight() * 0.725 - 64 * 2);
		square_plus.draw(ofGetWidth() * 0.95 - 64 * 3.3, ofGetHeight() * 0.725 - 64 * 2);
		square_plus.draw(ofGetWidth() * 0.95 - 64 * 4.45, ofGetHeight() * 0.725 - 64 * 2);

	}

	
}

void ofApp::drawImageLibrary() {
	float imageHeight = ofGetHeight()*0.5, imageWidth;
	ofImage image;

	if (imageDir.size() > 0) {
		
		image = imgLib[currentLibImage];
		currentImage = indexMap[currentLibImage];

		imageWidth = image.getWidth() * imageHeight / image.getHeight();
		image.resize(imageWidth, imageHeight);
		image.draw(ofGetWidth() * 0.35 - imageWidth/2, ofGetHeight() * 0.1);


	}

	previousIcon.draw(ofGetWidth()*0.35 - 64 * 2 - 20, ofGetHeight() * 0.65);
	nextIcon.draw(ofGetWidth()*0.35 + 64 + 20, ofGetHeight() * 0.65);
	homeIcon.draw(ofGetWidth() * 0.05, ofGetHeight() * 0.85);
	plusIcon.draw(ofGetWidth() * 0.95 - 64, ofGetHeight() * 0.85);

	if (plusImageToggle) {
		grid.draw(ofGetWidth() * 0.95 - grid.getWidth(), ofGetHeight() * 0.725 - grid.getHeight() / 2 - 64 / 1.7);

		square_plus.draw(ofGetWidth() * 0.95 - 64, ofGetHeight() * 0.725);
		square_plus.draw(ofGetWidth() * 0.95 - 64 * 2.15, ofGetHeight() * 0.725);
		square_plus.draw(ofGetWidth() * 0.95 - 64 * 3.3, ofGetHeight() * 0.725);
		square_plus.draw(ofGetWidth() * 0.95 - 64 * 4.45, ofGetHeight() * 0.725);

		square_plus.draw(ofGetWidth() * 0.95 - 64, ofGetHeight() * 0.725 - 64);
		square_plus.draw(ofGetWidth() * 0.95 - 64 * 2.15, ofGetHeight() * 0.725 - 64);
		square_plus.draw(ofGetWidth() * 0.95 - 64 * 3.3, ofGetHeight() * 0.725 - 64);
		square_plus.draw(ofGetWidth() * 0.95 - 64 * 4.45, ofGetHeight() * 0.725 - 64);

		square_plus.draw(ofGetWidth() * 0.95 - 64, ofGetHeight() * 0.725 - 64 * 2);
		square_plus.draw(ofGetWidth() * 0.95 - 64 * 2.15, ofGetHeight() * 0.725 - 64 * 2);
		square_plus.draw(ofGetWidth() * 0.95 - 64 * 3.3, ofGetHeight() * 0.725 - 64 * 2);
		square_plus.draw(ofGetWidth() * 0.95 - 64 * 4.45, ofGetHeight() * 0.725 - 64 * 2);

	}
}

void ofApp::reorderImgLib() {
	std::cout << "reordering" << std::endl;
	
	//imgs
	currentLibImage = 0;
	map<int, int> indexPoints;
	imgLib = images;
	//attribute points to each image based on the tags
	for (int i = 0; i < imgLib.size(); i++) {

		ofImage img = imgLib[i];
		XML.clear();
		XML.load("imgXmls/" + imageDir.getName(i) + ".xml");
		XML.pushTag("root");
		XML.pushTag("tags");
		vector<string> tags;
		bool cont = true;
		int j = 0;
		while (cont) {
			j++;
			string tagvalue = "tag" + std::to_string(j);
			string tag = XML.getValue(tagvalue, "");

			if (tag != "") {
				tags.push_back(tag);
			}
			else {
				cont = false;
			}
		}

		XML.popTag();
		XML.popTag();
		XML.clear();

		int points = 0;
		for (string tag : tags) {
			if (tagInVector(tag)) {
				points++;
			}
		}

		indexPoints[i] = points;
	}

	vector<int> sortedIndices(imgLib.size());
	std::iota(sortedIndices.begin(), sortedIndices.end(), 0);

	std::sort(sortedIndices.begin(), sortedIndices.end(), [&](int a, int b) {
		return indexPoints[a] > indexPoints[b];
		});

	indexMap.clear();
	for (int newIndex = 0; newIndex < sortedIndices.size(); newIndex++) {
		int originalIndex = sortedIndices[newIndex];
		indexMap[newIndex] = originalIndex;
	}

	vector<ofImage> sortedImages(imgLib.size());
	for (int newIndex = 0; newIndex < sortedIndices.size(); newIndex++) {
		int originalIndex = sortedIndices[newIndex];
		sortedImages[newIndex] = imgLib[originalIndex];
	}

	imgLib = sortedImages;

	//vids
	currentVideo = 0;
	map<int, int> indexPointsVid;
	vector<ofVideoPlayer> vidLib = videos;

}

void ofApp::drawWorkspace() {

	float x = ofGetWidth() * 0.085;
	float y = ofGetHeight() * 0.1;

	drawRow(x, y, 0);

	x = ofGetWidth() * 0.085;
	y = ofGetHeight() * 0.35;

	drawRow(x, y,4);

	x = ofGetWidth() * 0.085;
	y = ofGetHeight() * 0.6;

	drawRow(x, y,8);

	if (!workspaceMenuToggle) {
		arrowUpIcon.draw(ofGetWidth() * 0.05, ofGetHeight() * 0.85);
		deleteIcon.draw(ofGetWidth() * 0.95 - 64, ofGetHeight() * 0.85);
	}
	else {
		deleteIcon.draw(ofGetWidth() * 0.95 - 64, ofGetHeight() * 0.85);
		arrowDownIcon.draw(ofGetWidth() * 0.05, ofGetHeight() * 0.85);
		plusMenu.draw(ofGetWidth() * 0.05 - 12.5, ofGetHeight() * 0.725 - 64 * 2 - 20 * 2 - 12.5);
		imageIcon.draw(ofGetWidth() * 0.05, ofGetHeight() * 0.725 - 64 * 2 - 20 * 2);
		videoIcon.draw(ofGetWidth() * 0.05, ofGetHeight() * 0.725 - 64 - 20);
		facialDetectionIcon.draw(ofGetWidth() * 0.05, ofGetHeight() * 0.725);
	}

	if (trashToggle) {
		grid.draw(ofGetWidth() * 0.95 - grid.getWidth(), ofGetHeight() * 0.725 - grid.getHeight() / 2 - 64 / 1.7);

		square_plus.draw(ofGetWidth() * 0.95 - 64, ofGetHeight() * 0.725);
		square_plus.draw(ofGetWidth() * 0.95 - 64 * 2.15, ofGetHeight() * 0.725);
		square_plus.draw(ofGetWidth() * 0.95 - 64 * 3.3, ofGetHeight() * 0.725);
		square_plus.draw(ofGetWidth() * 0.95 - 64 * 4.45, ofGetHeight() * 0.725);

		square_plus.draw(ofGetWidth() * 0.95 - 64, ofGetHeight() * 0.725 - 64);
		square_plus.draw(ofGetWidth() * 0.95 - 64 * 2.15, ofGetHeight() * 0.725 - 64);
		square_plus.draw(ofGetWidth() * 0.95 - 64 * 3.3, ofGetHeight() * 0.725 - 64);
		square_plus.draw(ofGetWidth() * 0.95 - 64 * 4.45, ofGetHeight() * 0.725 - 64);

		square_plus.draw(ofGetWidth() * 0.95 - 64, ofGetHeight() * 0.725 - 64 * 2);
		square_plus.draw(ofGetWidth() * 0.95 - 64 * 2.15, ofGetHeight() * 0.725 - 64 * 2);
		square_plus.draw(ofGetWidth() * 0.95 - 64 * 3.3, ofGetHeight() * 0.725 - 64 * 2);
		square_plus.draw(ofGetWidth() * 0.95 - 64 * 4.45, ofGetHeight() * 0.725 - 64 * 2);

	}
 
}

void ofApp::drawRow(float x, float y, int offset) {
	float widthRatio, heightRatio, ratio, width, height;
	float canvasSlotWidth = ofGetWidth() * 0.2;
	float canvasSlotHeight = ofGetHeight() * 0.2;
	int type, index;

	ofImage image;
	ofVideoPlayer video;

	for (int i = 0+offset; i < 4+offset; i++) {
		
		type = canvasSlots[i][0];
		index = canvasSlots[i][1];

		if (type != EMPTY) {

			if (type == IMAGE) {
				image = images[index];

				widthRatio = canvasSlotWidth / image.getWidth();
				heightRatio = canvasSlotHeight / image.getHeight();

				if (widthRatio < heightRatio) {
					ratio = widthRatio;
				}
				else {
					ratio = heightRatio;
				}


				width = image.getWidth() * ratio;
				height = image.getHeight() * ratio;



				image.resize(width, height);
				image.draw(x + (canvasSlotWidth - width) / 2, y);
			}
			else if (type == VIDEO) {
				video = videos[index];

				widthRatio = canvasSlotWidth / video.getWidth();
				heightRatio = canvasSlotHeight / video.getHeight();

				if (widthRatio < heightRatio) {
					ratio = widthRatio;
				}
				else {
					ratio = heightRatio;
				}


				width = video.getWidth() * ratio;
				height = video.getHeight() * ratio;


				video.draw(x + (canvasSlotWidth - width) / 2, y, width, height);
				
			}

		}

		x += canvasSlotWidth + ofGetWidth() * 0.01;
	}

}

void ofApp::drawStandByScreen() {
	standByImage.resize(ofGetWidth(), ofGetHeight());
	standByImage.draw(0, 0);
}

void ofApp::drawFacialDetection() {
	float grabberX = ofGetWidth() / 2 - grabber.getWidth() / 2;
	float grabberY = ofGetHeight() * 0.1;

	grabber.draw(grabberX, grabberY);

	//draw the detected objects on top of the webcam image 
	ofNoFill();
	ofSetColor(255, 0, 255);
	for (auto res : results) {
		auto rect = res.rect;
		float x = rect.getX();
		float y = rect.getY();
		rect.setPosition(x + grabberX, y + grabberY);
		ofDrawRectangle(rect);
		ofDrawBitmapStringHighlight(res.label, rect.getTopLeft());
	}
	ofSetColor(255, 255, 255);

	homeIcon.draw(ofGetWidth() * 0.05, ofGetHeight() * 0.85);
	personIcon.draw(grabberX + grabber.getWidth()/2 - 20, grabberY + grabber.getHeight() + 20);
	font.drawString(ofToString(peopleNum), grabberX + grabber.getWidth() / 2 + 20, grabberY + grabber.getHeight() + 45);
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){
	switch(key) {
		case 'c': {
			if (cameraToggle) {
				currentWindow = previousWindow;
				cameraToggle = false;
			}
			else if (currentWindow != FACIALDETECTION) {
				if (currentWindow == VIDEOLIBRARY) {
					videos[currentVideo].stop();
					isPlaying = false;
				}

				previousWindow = currentWindow;
				currentWindow = FACIALDETECTION;
				cameraToggle = true;



			}
			break;
		}
		case 'f': {
			if (fullImageToggle) {
				currentWindow = IMAGELIBRARY;
				fullImageToggle = false;
			}
			else if(currentWindow == IMAGELIBRARY) {
				currentWindow = FULLSCREENIMAGE;
				fullImageToggle = true;
			}

			if (fullVideoToggle) {
				isPlaying = videoState;
				currentWindow = VIDEOLIBRARY;
				fullVideoToggle = false;
			}
			else if (currentWindow == VIDEOLIBRARY) {
				videoState = isPlaying;
				currentWindow = FULLSCREENVIDEO;
				fullVideoToggle = true;

			}

		}
	}
	
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y) {
	
}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){
	

	switch (currentWindow) {
		case 0: {
			break;
		}
		case 1: {

			if (!workspaceMenuToggle) {
				if (ofGetHeight() * 0.85 < y && y < ofGetHeight() * 0.85 + 64) {
					if (ofGetWidth() * 0.05 < x && x < ofGetWidth() * 0.05 + 64) {

						//arrow up
						workspaceMenuToggle = true;

					}
					
				}
			}
			else {
				if (ofGetWidth() * 0.05 < x && x < ofGetWidth() * 0.05 + 64) {

					if (ofGetHeight() * 0.725 < y && y < ofGetHeight() * 0.725 + 64) {
						//facial dectetion icon
						currentWindow = FACIALDETECTION;
						workspaceMenuToggle = false;
						trashToggle = false;
					}
					else if (ofGetHeight() * 0.725 - 64 - 20 < y && y < ofGetHeight() * 0.725 - 20) {
						//video icon
						currentWindow = VIDEOLIBRARY;
						workspaceMenuToggle = false;
						trashToggle = false;
					}
					else if (ofGetHeight() * 0.725 - 64 * 2 - 20 * 2 < y && y < ofGetHeight() * 0.725 - 64 - 20 * 2) {
						//image icon
						currentWindow = IMAGELIBRARY;
						workspaceMenuToggle = false;
						trashToggle = false;
					}
					else if (ofGetWidth() * 0.05 < x && x < ofGetWidth() * 0.05 + 64) {
						if (ofGetHeight() * 0.85 < y && y < ofGetHeight() * 0.85 + 64) {
							//arrow down
							workspaceMenuToggle = false;
						}
					}
				}
			}

			if (ofGetWidth() * 0.95 - 64 < x && x < ofGetWidth() * 0.95) {
				if (ofGetHeight() * 0.85 < y && y < ofGetHeight() * 0.85 + 64) {
					//trash icon
					trashToggle = !trashToggle;
				}
			}

			if (trashToggle) {
				if (ofGetWidth() * 0.95 - 64 * 4.45 < x && x < ofGetWidth() * 0.95 - 64 * 4.45 + square_plus.getWidth()) {
					if (ofGetHeight() * 0.725 - 64 * 2 < y && y < ofGetHeight() * 0.725 - 64 * 2 + square_plus.getHeight()) {
						//canvas slot 0
						canvasSlots[0][0] = EMPTY;
						canvasSlots[0][1] = 0;

					}
					else if (ofGetHeight() * 0.725 - 64 < y && y < ofGetHeight() * 0.725 - 64 + square_plus.getHeight()) {
						//canvas slot 4
						canvasSlots[4][0] = EMPTY;
						canvasSlots[4][1] = 0;
					}
					else if (ofGetHeight() * 0.725 < y && y < ofGetHeight() * 0.725 + square_plus.getHeight()) {
						//canvas slot 8
						canvasSlots[8][0] = EMPTY;
						canvasSlots[8][1] = 0;
					}

				}
				else if (ofGetWidth() * 0.95 - 64 * 3.3 < x && x < ofGetWidth() * 0.95 - 64 * 3.3 + square_plus.getWidth()) {
					if (ofGetHeight() * 0.725 - 64 * 2 < y && y < ofGetHeight() * 0.725 - 64 * 2 + square_plus.getHeight()) {
						//canvas slot 1
						canvasSlots[1][0] = EMPTY;
						canvasSlots[1][1] = 0;
					}
					else if (ofGetHeight() * 0.725 - 64 < y && y < ofGetHeight() * 0.725 - 64 + square_plus.getHeight()) {
						//canvas slot 5
						canvasSlots[5][0] = EMPTY;
						canvasSlots[5][1] = 0;
					}
					else if (ofGetHeight() * 0.725 < y && y < ofGetHeight() * 0.725 + square_plus.getHeight()) {
						//canvas slot 9
						canvasSlots[9][0] = EMPTY;
						canvasSlots[9][1] = 0;
					}
				}
				else if (ofGetWidth() * 0.95 - 64 * 2.15 < x && x < ofGetWidth() * 0.95 - 64 * 2.15 + square_plus.getWidth()) {
					if (ofGetHeight() * 0.725 - 64 * 2 < y && y < ofGetHeight() * 0.725 - 64 * 2 + square_plus.getHeight()) {
						//canvas slot 2
						canvasSlots[2][0] = EMPTY;
						canvasSlots[2][1] = 0;
					}
					else if (ofGetHeight() * 0.725 - 64 < y && y < ofGetHeight() * 0.725 - 64 + square_plus.getHeight()) {
						//canvas slot 6
						canvasSlots[6][0] = EMPTY;
						canvasSlots[6][1] = 0;
					}
					else if (ofGetHeight() * 0.725 < y && y < ofGetHeight() * 0.725 + square_plus.getHeight()) {
						//canvas slot 10
						canvasSlots[10][0] = EMPTY;
						canvasSlots[10][1] = 0;
					}

				}
				else if (ofGetWidth() * 0.95 - 64 < x && x < ofGetWidth() * 0.95 - 64 + square_plus.getWidth()) {
					if (ofGetHeight() * 0.725 - 64 * 2 < y && y < ofGetHeight() * 0.725 - 64 * 2 + square_plus.getHeight()) {
						//canvas slot 3
						canvasSlots[3][0] = EMPTY;
						canvasSlots[3][1] = 0;
					}
					else if (ofGetHeight() * 0.725 - 64 < y && y < ofGetHeight() * 0.725 - 64 + square_plus.getHeight()) {
						//canvas slot 7
						canvasSlots[7][0] = EMPTY;
						canvasSlots[7][1] = 0;
					}
					else if (ofGetHeight() * 0.725 < y && y < ofGetHeight() * 0.725 + square_plus.getHeight()) {
						//canvas slot 11
						canvasSlots[11][0] = EMPTY;
						canvasSlots[11][1] = 0;
					}

				}
			}
			
			break;
		}
		case 2: {
			if (ofGetWidth() * 0.05 < x && x < ofGetWidth() * 0.05 + 64) {
				if (ofGetHeight() * 0.85 < y && y < ofGetHeight() * 0.85 + 64) {
					//home icon
					currentWindow = WORKSPACE;
				}
			}
			break;
		}
		case 3: {
			if (ofGetHeight() * 0.85 < y && y < ofGetHeight() * 0.85 + 64) {
				if (ofGetWidth() * 0.05 < x && x < ofGetWidth() * 0.05 + 64) {
					//home icon 
					currentWindow = WORKSPACE;
				}
				else if (ofGetWidth() * 0.95 - 64 < x && x < ofGetWidth() * 0.95) {
					//plus icon
					plusImageToggle = !plusImageToggle;
				}
			}
			else if (ofGetHeight() * 0.65 < y && y < ofGetHeight() * 0.65 + 64) {
				if (ofGetWidth() * 0.35 - 64*23 - 20 < x && x < ofGetWidth() * 0.35 - 64 -20) {
					//previous icon 
					if (imageDir.size() > 0) {
						if (currentLibImage == 0)
							currentLibImage = imageDir.size();
						currentLibImage--;
					}


				}
				else if (ofGetWidth() * 0.35 + 64 + 20 < x && x < ofGetWidth() * 0.35 + 64*2 + 20) {
					//next icon
					if (imageDir.size() > 0) {
						currentLibImage++;
						currentLibImage %= imageDir.size();
					}
				}
			}

			if (plusImageToggle) {
				if (ofGetWidth() * 0.95 - 64 * 4.45 < x && x < ofGetWidth() * 0.95 - 64 * 4.45 + square_plus.getWidth()) {
					if (ofGetHeight() * 0.725 - 64 * 2 < y && y < ofGetHeight() * 0.725 - 64 * 2 + square_plus.getHeight()) {
						//canvas slot 0
						canvasSlots[0][0] = IMAGE;
						canvasSlots[0][1] = currentImage;
						plusImageToggle = false;
						currentWindow = WORKSPACE;
						addXmlTags(currentImage, true);
						reorderImgLib();

					}
					else if (ofGetHeight() * 0.725 - 64 < y && y < ofGetHeight() * 0.725 - 64 + square_plus.getHeight()) {
						//canvas slot 4
						canvasSlots[4][0] = IMAGE;
						canvasSlots[4][1] = currentImage;
						plusImageToggle = false;
						currentWindow = WORKSPACE;
						addXmlTags(currentImage, true);
						reorderImgLib();
					}
					else if (ofGetHeight() * 0.725 < y && y < ofGetHeight() * 0.725 + square_plus.getHeight()) {
						//canvas slot 8
						canvasSlots[8][0] = IMAGE;
						canvasSlots[8][1] = currentImage;
						plusImageToggle = false;
						currentWindow = WORKSPACE;
						addXmlTags(currentImage, true);
						reorderImgLib();

					}

				}
				else if (ofGetWidth() * 0.95 - 64 * 3.3 < x && x < ofGetWidth() * 0.95 - 64 * 3.3 + square_plus.getWidth()) {
					if (ofGetHeight() * 0.725 - 64 * 2 < y && y < ofGetHeight() * 0.725 - 64 * 2 + square_plus.getHeight()) {
						//canvas slot 1
						canvasSlots[1][0] = IMAGE;
						canvasSlots[1][1] = currentImage;
						plusImageToggle = false;
						currentWindow = WORKSPACE;
						addXmlTags(currentImage, true);
						reorderImgLib();

					}
					else if (ofGetHeight() * 0.725 - 64 < y && y < ofGetHeight() * 0.725 - 64 + square_plus.getHeight()) {
						//canvas slot 5
						canvasSlots[5][0] = IMAGE;
						canvasSlots[5][1] = currentImage;
						plusImageToggle = false;
						currentWindow = WORKSPACE;
						addXmlTags(currentImage, true);
						reorderImgLib();

					}
					else if (ofGetHeight() * 0.725 < y && y < ofGetHeight() * 0.725 + square_plus.getHeight()) {
						//canvas slot 9
						canvasSlots[9][0] = IMAGE;
						canvasSlots[9][1] = currentImage;
						plusImageToggle = false;
						currentWindow = WORKSPACE;
						addXmlTags(currentImage, true);
						reorderImgLib();

					}
				}
				else if (ofGetWidth() * 0.95 - 64 * 2.15 < x && x < ofGetWidth() * 0.95 - 64 * 2.15 + square_plus.getWidth()) {
					if (ofGetHeight() * 0.725 - 64 * 2 < y && y < ofGetHeight() * 0.725 - 64 * 2 + square_plus.getHeight()) {
						//canvas slot 2
						canvasSlots[2][0] = IMAGE;
						canvasSlots[2][1] = currentImage;
						plusImageToggle = false;
						currentWindow = WORKSPACE;
						addXmlTags(currentImage, true);
						reorderImgLib();

					}
					else if (ofGetHeight() * 0.725 - 64 < y && y < ofGetHeight() * 0.725 - 64 + square_plus.getHeight()) {
						//canvas slot 6
						canvasSlots[6][0] = IMAGE;
						canvasSlots[6][1] = currentImage;
						plusImageToggle = false;
						currentWindow = WORKSPACE;
						addXmlTags(currentImage, true);
						reorderImgLib();

					}
					else if (ofGetHeight() * 0.725 < y && y < ofGetHeight() * 0.725 + square_plus.getHeight()) {
						//canvas slot 10
						canvasSlots[10][0] = IMAGE;
						canvasSlots[10][1] = currentImage;
						plusImageToggle = false;
						currentWindow = WORKSPACE;
						addXmlTags(currentImage, true);
						reorderImgLib();

					}

				}
				else if (ofGetWidth() * 0.95 - 64 < x && x < ofGetWidth() * 0.95 - 64 + square_plus.getWidth()) {
					if (ofGetHeight() * 0.725 - 64 * 2 < y && y < ofGetHeight() * 0.725 - 64 * 2 + square_plus.getHeight()) {
						//canvas slot 3
						canvasSlots[3][0] = IMAGE;
						canvasSlots[3][1] = currentImage;
						plusImageToggle = false;
						currentWindow = WORKSPACE;
						addXmlTags(currentImage, true);
						reorderImgLib();

					}
					else if (ofGetHeight() * 0.725 - 64 < y && y < ofGetHeight() * 0.725 - 64 + square_plus.getHeight()) {
						//canvas slot 7
						canvasSlots[7][0] = IMAGE;
						canvasSlots[7][1] = currentImage;
						plusImageToggle = false;
						currentWindow = WORKSPACE;
						addXmlTags(currentImage, true);
						reorderImgLib();

					}
					else if (ofGetHeight() * 0.725 < y && y < ofGetHeight() * 0.725 + square_plus.getHeight()) {
						//canvas slot 11
						canvasSlots[11][0] = IMAGE;
						canvasSlots[11][1] = currentImage;
						plusImageToggle = false;
						currentWindow = WORKSPACE;
						addXmlTags(currentImage, true);
						reorderImgLib();

					}

				}

			}

			break;
		}
		case 4: {
			if (ofGetHeight() * 0.85 < y && y < ofGetHeight() * 0.85 + 64) {
				if (ofGetWidth() * 0.05 < x && x < ofGetWidth() * 0.05 + 64) {
					//home icon 
					videos[currentVideo].stop();
					isPlaying = false;
					currentWindow = WORKSPACE;
				}
				else if (ofGetWidth() * 0.95 - 64 < x && x < ofGetWidth() * 0.95) {
					//plus icon
					plusVideoToggle = !plusVideoToggle;
				}
			}
			else if (ofGetHeight() * 0.65 < y && y < ofGetHeight() * 0.65 + 64) {
				if (ofGetWidth() * 0.35 - 64 * 23 - 20 < x && x < ofGetWidth() * 0.35 - 64 - 20) {
					//previous icon 
					if (videoDir.size() > 0) {
						isPlaying = false;
						videos[currentVideo].stop();
						if (currentVideo == 0)
							currentVideo = videoDir.size();
						currentVideo--;
						
					}
					

				}
				else if (ofGetWidth() * 0.35 + 64 + 20 < x && x < ofGetWidth() * 0.35 + 64 * 2 + 20) {
					//next icon
					if (videoDir.size() > 0) {
						isPlaying = false;
						videos[currentVideo].stop();
						currentVideo++;
						currentVideo %= videoDir.size();
						
					}
				}
				else if (ofGetWidth() * 0.35 - 64/2 < x && x < ofGetWidth() * 0.35 + 64/2 + 64) {
					//play/pause icon
					if (videoDir.size() > 0) {
						if (isPlaying) {
							videos[currentVideo].setPaused(true);
						}
						else {
							videos[currentVideo].play();
						}
					}
					isPlaying = !isPlaying;
				}
			}

			if (plusVideoToggle) {
				if (ofGetWidth() * 0.95 - 64 * 4.45 < x && x < ofGetWidth() * 0.95 - 64 * 4.45 + square_plus.getWidth()) {
					if (ofGetHeight() * 0.725 - 64 * 2 < y && y < ofGetHeight() * 0.725 - 64 * 2 + square_plus.getHeight()) {
						//canvas slot 0
						canvasSlots[0][0] = VIDEO;
						canvasSlots[0][1] = currentVideo;
						plusVideoToggle = false;
						videos[currentVideo].stop();
						isPlaying = false;
						currentWindow = WORKSPACE;
						//addXmlTags(currentVideo, false);
					}
					else if (ofGetHeight() * 0.725 - 64 < y && y < ofGetHeight() * 0.725 - 64 + square_plus.getHeight()) {
						//canvas slot 4
						canvasSlots[4][0] = VIDEO;
						canvasSlots[4][1] = currentVideo;
						plusVideoToggle = false;
						videos[currentVideo].stop();
						isPlaying = false;
						currentWindow = WORKSPACE;
						//addXmlTags(currentVideo, false);
					}
					else if (ofGetHeight() * 0.725 < y && y < ofGetHeight() * 0.725 + square_plus.getHeight()) {
						//canvas slot 8
						canvasSlots[8][0] = VIDEO;
						canvasSlots[8][1] = currentVideo;
						plusVideoToggle = false;
						videos[currentVideo].stop();
						isPlaying = false;
						currentWindow = WORKSPACE;
						//addXmlTags(currentVideo, false);
					}

				}
				else if (ofGetWidth() * 0.95 - 64 * 3.3 < x && x < ofGetWidth() * 0.95 - 64 * 3.3 + square_plus.getWidth()) {
					if (ofGetHeight() * 0.725 - 64 * 2 < y && y < ofGetHeight() * 0.725 - 64 * 2 + square_plus.getHeight()) {
						//canvas slot 1
						canvasSlots[1][0] = VIDEO;
						canvasSlots[1][1] = currentVideo;
						plusVideoToggle = false;
						videos[currentVideo].stop();
						isPlaying = false;
						currentWindow = WORKSPACE;
						//addXmlTags(currentVideo, false);
					}
					else if (ofGetHeight() * 0.725 - 64 < y && y < ofGetHeight() * 0.725 - 64 + square_plus.getHeight()) {
						//canvas slot 5
						canvasSlots[5][0] = VIDEO;
						canvasSlots[5][1] = currentVideo;
						plusVideoToggle = false;
						videos[currentVideo].stop();
						isPlaying = false;
						currentWindow = WORKSPACE;
						//addXmlTags(currentVideo, false);

					}
					else if (ofGetHeight() * 0.725 < y && y < ofGetHeight() * 0.725 + square_plus.getHeight()) {
						//canvas slot 9
						canvasSlots[9][0] = VIDEO;
						canvasSlots[9][1] = currentVideo;
						plusVideoToggle = false;
						videos[currentVideo].stop();
						isPlaying = false;
						currentWindow = WORKSPACE;
						//addXmlTags(currentVideo, false);

					}
				}
				else if (ofGetWidth() * 0.95 - 64 * 2.15 < x && x < ofGetWidth() * 0.95 - 64 * 2.15 + square_plus.getWidth()) {
					if (ofGetHeight() * 0.725 - 64 * 2 < y && y < ofGetHeight() * 0.725 - 64 * 2 + square_plus.getHeight()) {
						//canvas slot 2
						canvasSlots[2][0] = VIDEO;
						canvasSlots[2][1] = currentVideo;
						plusVideoToggle = false;
						videos[currentVideo].stop();
						isPlaying = false;
						currentWindow = WORKSPACE;
						//addXmlTags(currentVideo, false);

					}
					else if (ofGetHeight() * 0.725 - 64 < y && y < ofGetHeight() * 0.725 - 64 + square_plus.getHeight()) {
						//canvas slot 6
						canvasSlots[6][0] = VIDEO;
						canvasSlots[6][1] = currentVideo;
						plusVideoToggle = false;
						videos[currentVideo].stop();
						isPlaying = false;
						currentWindow = WORKSPACE;
						//addXmlTags(currentVideo, false);

					}
					else if (ofGetHeight() * 0.725 < y && y < ofGetHeight() * 0.725 + square_plus.getHeight()) {
						//canvas slot 10
						canvasSlots[10][0] = VIDEO;
						canvasSlots[10][1] = currentVideo;
						plusVideoToggle = false;
						videos[currentVideo].stop();
						isPlaying = false;
						currentWindow = WORKSPACE;
						//addXmlTags(currentVideo, false);

					}

				}
				else if (ofGetWidth() * 0.95 - 64 < x && x < ofGetWidth() * 0.95 - 64 + square_plus.getWidth()) {
					if (ofGetHeight() * 0.725 - 64 * 2 < y && y < ofGetHeight() * 0.725 - 64 * 2 + square_plus.getHeight()) {
						//canvas slot 9
						canvasSlots[3][0] = VIDEO;
						canvasSlots[3][1] = currentVideo;
						plusVideoToggle = false;
						videos[currentVideo].stop();
						isPlaying = false;
						currentWindow = WORKSPACE;
						//addXmlTags(currentVideo, false);

					}
					else if (ofGetHeight() * 0.725 - 64 < y && y < ofGetHeight() * 0.725 - 64 + square_plus.getHeight()) {
						//canvas slot 10
						canvasSlots[7][0] = VIDEO;
						canvasSlots[7][1] = currentVideo;
						plusVideoToggle = false;
						videos[currentVideo].stop();
						isPlaying = false;
						currentWindow = WORKSPACE;
						//addXmlTags(currentVideo, false);

					}
					else if (ofGetHeight() * 0.725 < y && y < ofGetHeight() * 0.725 + square_plus.getHeight()) {
						//canvas slot 11
						canvasSlots[11][0] = VIDEO;
						canvasSlots[11][1] = currentVideo;
						plusVideoToggle = false;
						videos[currentVideo].stop();
						isPlaying = false;
						currentWindow = WORKSPACE;
						//saddXmlTags(currentVideo, false);

					}

				}
			}
			break;
		}
	}
}


//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){
}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){ 

}
