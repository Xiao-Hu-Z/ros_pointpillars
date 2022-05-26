# include "point_pillars_ros.h"

#define checkCudaErrors(status)                                   \
{                                                                 \
  if (status != 0)                                                \
  {                                                               \
    std::cout << "Cuda failure: " << cudaGetErrorString(status)   \
              << " at line " << __LINE__                          \
              << " in file " << __FILE__                          \
              << " error status: " << status                      \
              << std::endl;                                       \
              abort();                                            \
    }                                                             \
}

PointPillarsROS::PointPillarsROS()
  : nh_("~")
  , NUM_POINT_FEATURE_(4)
{
  Model_File = "../model/pointpillar.onnx";
}

void PointPillarsROS::createROSPubSub()
{
  sub_points_ = nh_.subscribe<sensor_msgs::PointCloud2>("/kitti/velo/pointcloud", 1, &PointPillarsROS::pointsCallback, this);
  pub_in_cloud = nh_.advertise<sensor_msgs::PointCloud2>("/kitti/in_points", 1);
  pub_objects_ = nh_.advertise<autoware_msgs::DetectedObjectArray>("/detection/lidar_detector/objects", 1);
  pub_markers_ =  nh_.advertise<visualization_msgs::MarkerArray>("/kitti/detection/visualize/cluster_markers", 1);
}


void pclToBin(const pcl::PointCloud<pcl::PointXYZI>::Ptr inPtr,void **data)
{
  char* points =new char[inPtr->points.size()*4];
  int  k=0;
  for(int i=0;i<inPtr->points.size();i++)
  {
    points[k] = inPtr->points[i].x;
    points[k+1] = inPtr->points[i].y;
    points[k+2] = inPtr->points[i].z;
    points[k+3] = inPtr->points[i].intensity;
    k += 4;
  }

  *data = (void*)points;
}

void publishCloud(
    ros::Publisher* in_publisher, std_msgs::Header header,
    pcl::PointCloud<pcl::PointXYZI>::Ptr inPtr) {
	sensor_msgs::PointCloud2 cloud_msg;
	pcl::toROSMsg(*inPtr, cloud_msg);
	cloud_msg.header = header;
	in_publisher->publish(cloud_msg);
}

void PointPillarsROS::pointsCallback(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
  pcl::PointCloud<pcl::PointXYZI>::Ptr inPtr(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::fromROSMsg(*msg, *inPtr);
  // pcl转bin
  void *data = NULL;
  std::shared_ptr<char> buffer((char *)data, std::default_delete<char[]>());
  pclToBin(inPtr,&data);
  buffer.reset((char *)data);
  float* points = (float*)buffer.get();

  // 创建一个事件，然后记录一个事件
  cudaEvent_t start, stop;
  float elapsedTime = 0.0f;
  cudaStream_t stream = NULL;

  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  checkCudaErrors(cudaStreamCreate(&stream));

  std::vector<Bndbox> nms_pred;
  nms_pred.reserve(100);
  PointPillar pointpillar(Model_File, stream);

  float *points_data = nullptr;
  unsigned int points_data_size = inPtr->size() * 4 * sizeof(float);
  checkCudaErrors(cudaMallocManaged((void **)&points_data, points_data_size));
  checkCudaErrors(cudaMemcpy(points_data, points, points_data_size, cudaMemcpyDefault));

  checkCudaErrors(cudaDeviceSynchronize());
  // 记录多GPU程序的CUDA内核的运行时间
  cudaEventRecord(start, stream);

  pointpillar.doinfer(points_data, inPtr->size(), nms_pred);
  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  std::cout<<"TIME: pointpillar: "<< elapsedTime <<" ms." <<std::endl;

  checkCudaErrors(cudaFree(points_data));

  std::cout<<"Bndbox objs: "<< nms_pred.size()<<std::endl;

  publishCloud(&pub_in_cloud,msg->header,inPtr);
  autoware_msgs::DetectedObjectArray detected_objects;
  pubDetectedObject(nms_pred,msg->header,detected_objects);

  // 可视化
  visualization_msgs::MarkerArray visualize_markers;
  vdo.visualizeDetectedObjs(detected_objects,visualize_markers);
  pub_markers_.publish(visualize_markers);
  nms_pred.clear();
  std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>" <<std::endl;
}


void PointPillarsROS::pubDetectedObject(const std::vector<Bndbox>& detections, const std_msgs::Header& in_header,autoware_msgs::DetectedObjectArray& objects)
{
  objects.header = in_header;
  for (size_t i = 0; i < detections.size(); i++)
  {
    autoware_msgs::DetectedObject object;
    object.header = in_header;
    object.valid = true;
    object.pose_reliable = true;
    
    object.dimensions.x = detections[i].l;
    object.dimensions.y = detections[i].w;
    object.dimensions.z = detections[i].h;

    object.pose.position.x = detections[i].x + object.dimensions.x/2;
    object.pose.position.y = detections[i].y + object.dimensions.y/2;
    object.pose.position.z = detections[i].z + object.dimensions.z/2;


    // Trained this way
    float yaw = detections[i].rt;
    yaw += M_PI/2;
    yaw = std::atan2(std::sin(yaw), std::cos(yaw));
    geometry_msgs::Quaternion q = tf::createQuaternionMsgFromYaw(-yaw);
    object.pose.orientation = q;

    //Only detects car in Version 1.0
    object.label = "car";

    objects.objects.push_back(object);
  }
  pub_objects_.publish(objects);
}