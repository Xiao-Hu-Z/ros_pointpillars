#include <iostream>

// headers in local files
#include "point_pillars_ros.h"

int main(int argc, char** argv)
{
  ros::init(argc, argv, "lidar_point_pillars");
  PointPillarsROS app;
  app.createROSPubSub();
  ros::spin();

  return 0;
}
