#pragma once

#include "ofMain.h"
#include "ofxOpenCv.h"
#include "yolo5ImageClassify.h"
#include "ofxXmlSettings.h"


class ofApp : public ofBaseApp{

	public:
		void setup();
		void update();
		void draw();
		void drawStandByScreen();
		void drawFacialDetection();
		void drawWorkspace();
		void drawImageLibrary();
		void drawVideoLibrary();
		void drawFullImage();
		void drawFullVideo();
		void keyPressed(int key);
		void keyReleased(int key);
		void mouseMoved(int x, int y );
		void mouseDragged(int x, int y, int button);
		void mousePressed(int x, int y, int button);
		void mouseReleased(int x, int y, int button);
		void mouseEntered(int x, int y);
		void mouseExited(int x, int y);
		void windowResized(int w, int h);
		void dragEvent(ofDragInfo dragInfo);
		void gotMessage(ofMessage msg);
		void drawRow(float x, float y, int offset);
		void handleXMLDoesNotExist(ofImage img, int i);
		void handleXMLDoesNotExist(ofVideoPlayer vid, int i);
		void addXmlTags(int index, bool isImage);
		bool tagInVector(string tag);
		void reorderImgLib();//reorder the images in the image library based on the tags

		ofVideoGrabber grabber;
		ofxCvColorImage	colorImg;
		yolo5ImageClassify classify;
		vector <yolo5ImageClassify::Result> results;


		int currentWindow;
		int previousWindow;
		bool cameraToggle;
		bool fullImageToggle;
		bool fullVideoToggle;
		bool workspaceMenuToggle;
		bool plusImageToggle;
		bool plusVideoToggle;
		bool trashToggle;

		bool videoState;
		int peopleNum;
		ofTrueTypeFont font;

		//canvas
		const int EMPTY = 0;
		const int IMAGE = 1;
		const int VIDEO = 2;
		const int SLOTNUM = 12;
		int canvasSlots[12][2]; //the canvas has 12 slots in total. type of object stored in the x index: [x][0]. index of the object in their directory: [x][1]

		//images
		ofDirectory imageDir;
		vector<ofImage> images;
		int currentImage;
		int currentLibImage;
		vector<ofImage> imgLib;
		map<int, int> indexMap;

		//videos
		ofDirectory videoDir;
		vector<ofVideoPlayer> videos;
		int currentVideo;
		int currentLibVideo;
		bool isPlaying;
		vector<ofVideoPlayer> vidLib;
		map<int, int> vidIndexMap;


		//ui
		ofImage plusIcon;
		ofImage deleteIcon;
		ofImage plusMenu;
		ofImage imageIcon;
		ofImage videoIcon;
		ofImage homeIcon;
		ofImage personIcon;
		ofImage previousIcon;
		ofImage nextIcon;
		ofImage playIcon;
		ofImage pauseIcon;
		ofImage facialDetectionIcon;
		ofImage standByImage;
		ofImage arrowUpIcon;
		ofImage arrowDownIcon;
		ofImage square_plus;
		ofImage grid;

		//screens
		const int STANDBY = 0;
		const int WORKSPACE = 1;
		const int FACIALDETECTION = 2;
		const int IMAGELIBRARY = 3;
		const int VIDEOLIBRARY = 4;
		const int FULLSCREENIMAGE = 5;
		const int FULLSCREENVIDEO = 6;
		
		ofxXmlSettings XML;
		vector<string> selectedTags;
};

