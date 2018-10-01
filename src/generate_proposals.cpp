#include <iostream>

#include<opencv2/opencv.hpp>
#include<pcl/common/common.h>
#include<pcl/filters/voxel_grid.h>
#include<pcl/filters/passthrough.h>
#include<pcl/filters/extract_indices.h>
#include<pcl/segmentation/sac_segmentation.h>
#include<pcl/segmentation/extract_clusters.h>
#include<pcl/visualization/cloud_viewer.h>
#include<dirent.h>

//#include<pcl/ModelCoefficients.h>
//#include<pcl/PointIndices.h>
//#include<pcl/point_cloud.h>

struct BboxTemplate
{
    BboxTemplate(double height_in, double width_in)
    {
        height = height_in;
        width = width_in;
    }
    double height;
    double width;
};

struct Bbox
{
    int xmin;
    int ymin;
    int xmax;
    int ymax;
};

struct PlaneCoordinates
{
    //components of the plane equation in point-normal form:
    //ax + by + cz + d = 0
    PlaneCoordinates()
    {
        valid = false;
    }
    double a;
    double b;
    double c;
    double d;
    bool valid;
};

struct CameraCalibration
{
    double cx;
    double cy;
    double fx;
    double fy;
};

class ProposalGenerator
{
public:
    ProposalGenerator();
    ProposalGenerator(CameraCalibration cam_calibration, double camera_height);
    std::vector<Bbox> get_proposals(cv::Mat depth_image);
private:
    void init(CameraCalibration cam_calibration, double camera_height);
    pcl::PointCloud<pcl::PointXYZ>::Ptr get_cloud(cv::Mat depth_image, int sample_factor);
    pcl::PointCloud<pcl::PointXYZ>::Ptr apply_voxel_filter(
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, double leaf_size);
    PlaneCoordinates estimate_ground_plane(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in);
    pcl::PointCloud<pcl::PointXYZ>::Ptr remove_ground_plane(
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in,
            PlaneCoordinates ground_plane);
    std::vector<pcl::PointIndices> cluster_pointcloud(
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in);

    bool find_cluster_center(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in,
                             pcl::PointIndices cluster_indeces,
                             double& center_x, double& center_z);

    std::vector<BboxTemplate> get_templates();

    CameraCalibration cam_calib_;
    double camera_height_;
};

ProposalGenerator::ProposalGenerator()
{
    //default camera calibration values:
    CameraCalibration cam_calibration;
    cam_calibration.fx = 540.686;
    cam_calibration.fy = 540.686;
    cam_calibration.cx = 479.75;
    cam_calibration.cy = 269.75;
    double camera_height = 1.1;
    this->init(cam_calibration, camera_height);
}

ProposalGenerator::ProposalGenerator(CameraCalibration cam_calibration, double camera_height)
{
    this->init(cam_calibration, camera_height);
}

void ProposalGenerator::init(CameraCalibration cam_calibration, double camera_height)
{
    cam_calib_ = cam_calibration;
    camera_height_ = camera_height;
}

//TODO input images must have depth values in meters
pcl::PointCloud<pcl::PointXYZ>::Ptr ProposalGenerator::get_cloud(
        cv::Mat depth_image, int sample_factor)
{
    //Convert image to point cloud
    //Sample every ith pixel in the image to build the point cloud

    int width_im = depth_image.cols;
    int height_im = depth_image.rows;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in (new pcl::PointCloud<pcl::PointXYZ>);
    cloud_in->width = int(width_im/sample_factor);
    cloud_in->height = int(height_im/sample_factor);
    cloud_in->is_dense = true; //all the data in cloud_in is finite (No Inf/NaN values)
    cloud_in->points.resize (cloud_in->width * cloud_in->height);

    int i=0;
    double max_depth = 0;
    double meter_factor = 1000.0;

    for (int x=0; x<width_im; x+=sample_factor)
    {
        for (int y=0; y<height_im; y+=sample_factor)
        {
            float depth = (float)depth_image.at<uint16_t>(y, x) / meter_factor;
            if(depth > max_depth)
                max_depth = depth;

            if (!std::isnan(depth))
            {
                float pclx = (x - cam_calib_.cx ) / cam_calib_.fx * depth;
                float pcly = (y - cam_calib_.cy ) / cam_calib_.fy * depth;

                cloud_in->points[i].x = pclx;
                cloud_in->points[i].y = pcly;
                cloud_in->points[i].z = depth;
            }
            i++;
        }
    }
//    std::cout << "max depth: " << max_depth << std::endl;
    return cloud_in;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr ProposalGenerator::apply_voxel_filter(
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, double leaf_size)
{
    //Voxel filter: Downsample the point cloud using a leaf size of 0.1mts
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setLeafSize (leaf_size, leaf_size, leaf_size);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_down (new pcl::PointCloud<pcl::PointXYZ>);
    vg.setInputCloud (cloud_in);
    vg.filter (*cloud_down);
    return cloud_down;
}

PlaneCoordinates ProposalGenerator::estimate_ground_plane(
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in)
{
    PlaneCoordinates plane_coordinates;

    // TODO this step assumes the camera stays at a fixed pose, it limits the performance
    // Keep points whose "y" coordinates are between "camera_height-above" and "camera_height+under"
    float above = 0.6; //meters above the expected ground plane
    float under = 0.2; //meters under the expected ground plane
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cropped (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud (cloud_in);
    pass.setFilterFieldName ("y");
    pass.setFilterLimits (camera_height_-above, camera_height_+under);
    pass.filter (*cloud_cropped);

    int min_cropped_cloud_points = 10;
    if (cloud_cropped->points.size() > min_cropped_cloud_points)
    {
        pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        seg.setOptimizeCoefficients (true);
        seg.setModelType (pcl::SACMODEL_PLANE);
        seg.setMethodType (pcl::SAC_RANSAC);
        seg.setDistanceThreshold (0.05);

        // Get inliers of the predicted plane
        seg.setInputCloud (cloud_cropped);

        // Estimate coefficients
        seg.segment (*inliers, *coefficients);

        float plane_max_tolerance = 0.2; //meters
        int min_num_inliers = 40;

        //estimated plane is valid only if it is close to the expected ground plane
        if(inliers->indices.size() >= min_num_inliers && abs(coefficients->values[3]-camera_height_) < plane_max_tolerance)
        {
            plane_coordinates.a = coefficients->values[0];
            plane_coordinates.b = coefficients->values[1];
            plane_coordinates.c = coefficients->values[2];
            plane_coordinates.d = coefficients->values[3];
            plane_coordinates.valid = true;
        }
    }
    return plane_coordinates;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr ProposalGenerator::remove_ground_plane(
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, PlaneCoordinates ground_plane)
{
    double floor_limit = 0.3;
    double ceiling_limit = 2.0;

    // Get index of points close to ground plane
    pcl::PointIndices index_cloud;
    for (size_t i = 0; i < cloud_in->points.size (); ++i)
    {
        //TODO instead of comparing the y coordinate, we should compare the
        //distance to plane to be robust against camera orientations
        double plane_y = -(ground_plane.a*cloud_in->points[i].x + ground_plane.c*cloud_in->points[i].z + ground_plane.d)/ground_plane.b;
        // Keep points above the ground plane and under the ceiling
        if(cloud_in->points[i].y < plane_y-floor_limit && cloud_in->points[i].y > plane_y-ceiling_limit){
            index_cloud.indices.push_back(i);}
    }

    // Remove ground plane: Remove points close to ground plane
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_without_plane (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::ExtractIndices<pcl::PointXYZ> eifilter (true);
    eifilter.setInputCloud (cloud_in);
    eifilter.setIndices (boost::make_shared<const pcl::PointIndices> (index_cloud));
    eifilter.filter (*cloud_without_plane);

    return cloud_without_plane;
}

std::vector<pcl::PointIndices> ProposalGenerator::cluster_pointcloud(
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in)
{
    // Creating the KdTree object for segmentation
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud (cloud_in);

    // Apply Euclidean Clustering
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance (0.12); //0.12 points separated less than 12cm are part of same cluster
    ec.setMinClusterSize (15);
    ec.setMaxClusterSize (1300);
    ec.setSearchMethod (tree);
    ec.setInputCloud (cloud_in);
    ec.extract (cluster_indices);

    return cluster_indices;
}

bool ProposalGenerator::find_cluster_center(
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in,
        pcl::PointIndices cluster_indeces, double &center_x, double &center_z)
{
    center_x = 0;
    center_z = 0;

    int count_cent = 0;

    if(cluster_indeces.indices.size() > 0)
    {
        //iterate over points in cluster to find centroid of cluster
        for (std::vector<int>::const_iterator pit = cluster_indeces.indices.begin();
             pit != cluster_indeces.indices.end(); ++pit)
        {
            center_x += cloud_in->points[*pit].x;
            center_z += cloud_in->points[*pit].z;
            count_cent++;
        }

        center_x /= count_cent;
        center_z /= count_cent;

        return true;
    }

    return false;
}

std::vector<BboxTemplate> ProposalGenerator::get_templates()
{
    double pedestrian_height = 1.75; //average pedestrian height
    double pedestrian_width = 0.6; //average pedestrian width

    std::vector<BboxTemplate> bbox_templates;
    //T1 template for pedestrians and crutches and walking frame front views
    bbox_templates.push_back(BboxTemplate(pedestrian_height, pedestrian_width));
    //T2 template for side views of crutches and walking frame
    bbox_templates.push_back(BboxTemplate(pedestrian_height, 5.0/3.0*pedestrian_width));
    //T3 template for side views of people pushing people with wheelchair
    bbox_templates.push_back(BboxTemplate(pedestrian_height, 7.0/3.0*pedestrian_width));
    //T4 template for front view of wheelchair
    bbox_templates.push_back(BboxTemplate(3.0/4.0*pedestrian_height, pedestrian_width));
    //T5 template for side view of wheelchair
    bbox_templates.push_back(BboxTemplate(3.0/4.0*pedestrian_height, 5.0/3.0*pedestrian_width));

    return bbox_templates;
}

using namespace cv;

// Function to apply JET color mapping to Depth images
void depth2jet(Mat& input_depth, Mat& colored_depth) {
    double min;
    double max;
    cv::minMaxIdx(input_depth, &min, &max);
    // expand your range to 0..255. Similar to histEq();
    input_depth.convertTo(colored_depth,CV_8UC1, 255 / (max-min), -min);
    applyColorMap(colored_depth, colored_depth, COLORMAP_JET);
}

std::vector<Bbox> ProposalGenerator::get_proposals(cv::Mat depth_image)
{
    std::vector<Bbox> proposals;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = this->get_cloud(depth_image, 4);
//    viewer.addPointCloud<pcl::PointXYZ>(cloud, "cloud_in", port1);
//    viewer.createViewPortCamera(port1);
//    std::cout << "showing input cloud " << std::endl;

    cloud = this->apply_voxel_filter(cloud, 0.1);

    PlaneCoordinates ground_plane = this->estimate_ground_plane(cloud);
    cloud = this->remove_ground_plane(cloud, ground_plane);

    //while (!viewer.wasStopped ()){}

    std::vector<pcl::PointIndices> clusters = this->cluster_pointcloud(cloud);

    std::vector<BboxTemplate> bbox_templates = get_templates();
    double stride = 0.1; //slide windows 10cm to each side

    //find center of each cluster and apply local sliding templates
    for(std::vector<pcl::PointIndices>::iterator cluster_it = clusters.begin();
        cluster_it!= clusters.end(); cluster_it++)
    {
        pcl::PointIndices cluster_indeces = *cluster_it;
        double center_x, center_z;

        if(this->find_cluster_center(cloud, cluster_indeces, center_x, center_z))
        {
            Bbox bbox;

//            cv::Mat depthjet_image;
//            depth2jet(depth_image, depthjet_image);
//            cv::Mat im_copy = depthjet_image;

//            double pnt_x, pnt_y;
//            for (std::vector<int>::const_iterator pit = cluster_indeces.indices.begin();
//                 pit != cluster_indeces.indices.end(); ++pit)
//            {
//                pnt_x = cloud->points[*pit].x * cam_calib_.fx/(cloud->points[*pit].z) + cam_calib_.cx;
//                pnt_y = cloud->points[*pit].y * cam_calib_.fy/(cloud->points[*pit].z) + cam_calib_.cy;
//                cv::circle(im_copy, Point(pnt_x, pnt_y), 5, Scalar(255, 255, 255),-1);
//            }

            //bbox y coordinates are determined by the plane and the template height
            double plane_y = -(ground_plane.a*center_x + ground_plane.c*center_z +ground_plane.d)/ground_plane.b;

            for(std::vector<BboxTemplate>::iterator template_it = bbox_templates.begin();
                template_it != bbox_templates.end(); template_it++)
            {
                BboxTemplate bbox_template = *template_it;

                bbox.ymax = plane_y * cam_calib_.fy/(center_z) + cam_calib_.cy;
                bbox.ymin = (plane_y - bbox_template.height) * cam_calib_.fy/center_z + cam_calib_.cy;

                //slide templates to left and right by the stride
                for(int kk=-1; kk<=1; kk++)
                {
                    double eval_x = center_x + kk * stride;
                    bbox.xmin = (eval_x - bbox_template.width/2.0) * cam_calib_.fx/center_z + cam_calib_.cx;
                    bbox.xmax = (eval_x + bbox_template.width/2.0) * cam_calib_.fx/center_z + cam_calib_.cx;

                    //clip to image boundaries
                    bbox.xmin = std::max(0, std::min(bbox.xmin, depth_image.cols));
                    bbox.xmax = std::max(0, std::min(bbox.xmax, depth_image.cols));
                    bbox.ymin = std::max(0, std::min(bbox.ymin, depth_image.rows));
                    bbox.ymax = std::max(0, std::min(bbox.ymax, depth_image.rows));

                    proposals.push_back(bbox);

//                    if(kk==-1)
//                        cv::rectangle(im_copy, cv::Point(bbox.xmax,bbox.ymax), Point(bbox.xmin,bbox.ymin), Scalar(255,0,0), 2, 8);
//                    if(kk==0)
//                        cv::rectangle(im_copy, cv::Point(bbox.xmax,bbox.ymax), Point(bbox.xmin,bbox.ymin), Scalar(255,255,255), 2, 8);
//                    if(kk==1)
//                        cv::rectangle(im_copy, cv::Point(bbox.xmax,bbox.ymax), Point(bbox.xmin,bbox.ymin), Scalar(0,0,255), 2, 8);
                }
//                cv::imshow( "Display window", im_copy);// Show our image inside it.
//                cv::waitKey(0);
            }
        }
    }

    return proposals;
}

int main (int argc, char** argv)
{
    ProposalGenerator prop_gen;

    std::string dataset_path = "/home/kollmitz/datasets/mobility-aids/";
    std::string depth_dir = dataset_path + "Depth/";
    std::string roidb_dir = dataset_path + "roidb_segmentation/";

    //find files in depth_dir
    std::vector<std::string> v;
    DIR* dirp = opendir(depth_dir.c_str());
    struct dirent * dp;
    while ((dp = readdir(dirp)) != NULL)
    {
        if(std::string(dp->d_name).find(".png") != std::string::npos)
        {
            v.push_back(dp->d_name);
        }
    }
    closedir(dirp);

    std::cout << "found " << v.size() << " Depth images. Generating roidb segmentation files..." << std::endl;

    for(int i=0; i<v.size(); i++)
    {
        std::string image_name = v.at(i);
        std::string image_path = depth_dir + image_name;
        cv::Mat depth_image = cv::imread(image_path, CV_16UC1);

        std::vector<Bbox> proposals = prop_gen.get_proposals(depth_image);

        std::cout << v.at(i) << ": found " << proposals.size() << " proposals" << std::endl;

        ofstream myfile;
        myfile.open((roidb_dir + image_name.substr(0, image_name.size()-4) + ".txt").c_str());

        for(int i=0; i<proposals.size(); i++)
        {
            Bbox proposal = proposals.at(i);
            myfile << proposal.xmin << " " << proposal.ymin << " " << proposal.xmax << " " << proposal.ymax << std::endl;
        }
        myfile.close();
    }

    std::cout << "done" << std::endl;
}
