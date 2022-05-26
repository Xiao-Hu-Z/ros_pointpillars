#ifndef POINTS_PILLAR_ROS_H
#define POINTS_PILLAR_ROS_H

#include <memory>
#include <vector>
#include <chrono>
#include <cmath>

#include "cuda_runtime.h"
#include <ros/ros.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl_ros/transforms.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_listener.h>
#include "autoware_msgs/DetectedObjectArray.h"

#include "params.h"
#include "pointpillar.h"
#include "visualize_detected_objects.h"

class PointPillarsROS
{
private:
  std::string Model_File;
  // 列表初始化
  const int NUM_POINT_FEATURE_;

  ros::NodeHandle nh_;
  ros::Subscriber sub_points_;

  ros::Publisher pub_in_cloud;
  ros::Publisher pub_objects_;
  ros::Publisher pub_markers_;

  VisualizeDetectedObjects vdo;

  tf::TransformListener tf_listener_;
  tf::StampedTransform baselink2lidar_;
  tf::Transform angle_transform_;
  tf::Transform angle_transform_inversed_;

  float offset_z_from_trained_data_;



  void pointsCallback(const sensor_msgs::PointCloud2::ConstPtr& input);
  void pubDetectedObject(const std::vector<Bndbox>& detections, const std_msgs::Header& in_header,autoware_msgs::DetectedObjectArray& objects);


public:
  PointPillarsROS();
  void createROSPubSub();
};

#endif  // POINTS_PILLAR_ROS_H
