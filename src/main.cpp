#include <iostream>
#include <string>
#include <pcl/console/parse.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>    
#include <pcl/io/openni_grabber.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/people/ground_based_people_detection_app.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <tiobj.hpp>
#include <unistd.h>


typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloudT;


using namespace std;
using namespace cv;


pcl::visualization::PCLVisualizer viewer("PCL Viewer");




int main(){
	float min_confidence = -1.5;
	float min_height = 1.3;
	float max_height = 2.3;
	float voxel_size = 0.06;

	float fx = 525;
	float fy = 525;
	float cx = 319.5;
	float cy = 239.5;

	// Kinect RGB camera intrinsics
	Eigen::Matrix3f rgb_intrinsics_matrix;
	rgb_intrinsics_matrix << 
		525, 0.0, 319.5,
		0.0, 525, 239.5,
		0.0, 0.0, 1.0; 



	PointCloudT::Ptr cloud( new PointCloudT(  0,0,PointT(0,0,0)  ) );


	FILE* fd;
	Mat depth;
	depth.create(Size(640,480), CV_16S);
	fd = fopen("depth","r");
	fread(depth.data, 2, 640*480, fd);
	fclose(fd);

	Mat image;
	image.create(Size(640,480), CV_8UC3);
	fd = fopen("image","r");
	fread(image.data, 3, 640*480, fd);
	fclose(fd);

	PointT p;
	for (int iy=0;iy<image.rows;iy++){
		for (int ix=0;ix<image.cols;ix++){
			int z = depth.at<short>(iy,ix);
			if ( z < 2047 ){
				Vec3b& a = image.at<Vec3b>(iy,ix);
				p = PointT(a[0],a[1],a[2]);
				p.x = (ix - cx) * z / fx;
				p.y = (iy - cy) * z / fy;
				p.z = z;
				cloud->push_back(p);
			}
		}
	}

	/*PointT p(20,30,40);
	cloud->push_back(p);
	cout << p << endl;*/

	pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb(cloud);
	viewer.addPointCloud<PointT> (cloud, rgb, "input_cloud");
	viewer.setCameraPosition(0,0,-2,0,-1,0,0);
	viewer.spin();


	/*pcl::people::PersonClassifier<pcl::RGB> person_classifier;
	person_classifier.loadSVMFromFile("../etc/trainedLinearSVMForPeopleDetectionWithHOG.yaml");



	pcl::people::GroundBasedPeopleDetectionApp<PointT> people_detector;    // people detection object
	people_detector.setVoxelSize(voxel_size);                        // set the voxel size
	people_detector.setIntrinsics(rgb_intrinsics_matrix);            // set RGB camera intrinsic parameters
	people_detector.setClassifier(person_classifier);                // set person classifier
	//people_detector.setHeightLimits(min_height, max_height);         // set person classifier*/


	return 0;
}
