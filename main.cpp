// /*
//  * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//  * SPDX-License-Identifier: Apache-2.0
//  *
//  * Licensed under the Apache License, Version 2.0 (the "License");
//  * you may not use this file except in compliance with the License.
//  * You may obtain a copy of the License at
//  *
//  * http://www.apache.org/licenses/LICENSE-2.0
//  *
//  * Unless required by applicable law or agreed to in writing, software
//  * distributed under the License is distributed on an "AS IS" BASIS,
//  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  * See the License for the specific language governing permissions and
//  * limitations under the License.
//  */

// #include <iostream>
// #include <sstream>
// #include <fstream>
// #include <dirent.h>
// //#include <iterator>

// #include <ros/ros.h>
// #include <sensor_msgs/PointCloud2.h>
// #include "cuda_runtime.h"

// #include "./params.h"
// #include "./pointpillar.h"

// #include "autoware_msgs/DetectedObjectArray.h"
// #include "visualize_detected_objects.h"
// #include <tf/transform_listener.h>

// #include <pcl_conversions/pcl_conversions.h>
// #include <pcl/PCLPointCloud2.h>
// #include <pcl/conversions.h>
// #include <pcl_ros/point_cloud.h>
// #include <pcl/point_types.h>

// #define checkCudaErrors(status)                                   \
// {                                                                 \
//   if (status != 0)                                                \
//   {                                                               \
//     std::cout << "Cuda failure: " << cudaGetErrorString(status)   \
//               << " at line " << __LINE__                          \
//               << " in file " << __FILE__                          \
//               << " error status: " << status                      \
//               << std::endl;                                       \
//               abort();                                            \
//     }                                                             \
// }

// std::string Data_File = "/media/xiaohu/xiaohu/new start/数据集/激光/object/testing/velodyne/";
// //std::string Data_File = "../data/";
// std::string Model_File = "../model/pointpillar.onnx";

// ros::Publisher pub_in_cloud;
// ros::Publisher pub_objects_;
// ros::Publisher publisher_markers_;
// ros::Publisher _pub_in_cloud;

// std_msgs::Header velodyneHeader;

// VisualizeDetectedObjects vdo;

// void Getinfo(void)
// {
//   cudaDeviceProp prop;

//   int count = 0;
//   cudaGetDeviceCount(&count);
//   printf("\nGPU has cuda devices: %d\n", count);
//   for (int i = 0; i < count; ++i) {
//     cudaGetDeviceProperties(&prop, i);
//     printf("----device id: %d info----\n", i);
//     printf("  GPU : %s \n", prop.name);
//     printf("  Capbility: %d.%d\n", prop.major, prop.minor);
//     printf("  Global memory: %luMB\n", prop.totalGlobalMem >> 20);
//     printf("  Const memory: %luKB\n", prop.totalConstMem  >> 10);
//     printf("  SM in a block: %luKB\n", prop.sharedMemPerBlock >> 10);
//     printf("  warp size: %d\n", prop.warpSize);
//     printf("  threads in a block: %d\n", prop.maxThreadsPerBlock);
//     printf("  block dim: (%d,%d,%d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
//     printf("  grid dim: (%d,%d,%d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
//   }
//   printf("\n");
// }

// int loadData(const char *file, void **data, unsigned int *length,pcl::PointCloud<pcl::PointXYZI>::Ptr &cloudPtr)
// {
//   std::fstream dataFile(file, std::ifstream::in);

//   if (!dataFile.is_open())
//   {
// 	  std::cout << "Can't open files: "<< file<<std::endl;
// 	  return -1;
//   }

//   //get length of file:
//   unsigned int len = 0;
//   // seekg 函数则用于已经打开要进行输入的文件
//   // seekg 则可用于从文件中 get（获取）信息
//   // ios::end	从文件末尾开始计算偏移量
//   dataFile.seekg (0, dataFile.end);
//   // tellp 用于返回写入位置，tellg 则用于返回读取位置
//   len = dataFile.tellg();
//   // ios::beg	从文件头开始计算偏移量
//   dataFile.seekg (0, dataFile.beg);
  
//   // bin转pcl
// 	for (int i=0; dataFile.good() && !dataFile.eof(); i++) {
// 		pcl::PointXYZI point;
// 		dataFile.read((char *) &point.x, 3*sizeof(float));
// 		dataFile.read((char *) &point.intensity, sizeof(float));
// 		cloudPtr->push_back(point);
// 	}

//   //allocate memory:
//   char *buffer = new char[len];
//   if(buffer==NULL) {
// 	  std::cout << "Can't malloc buffer."<<std::endl;
//     dataFile.close();
// 	  exit(-1);
//   }

//   //read data as a block:
//   dataFile.read(buffer, len);
//   dataFile.close();

//   *data = (void*)buffer;
//   *length = len;
//   return 0; 
// }

// void publishCloud(
//     ros::Publisher* in_publisher, std_msgs::Header header,
//     pcl::PointCloud<pcl::PointXYZI>::Ptr inPtr) {
// 	sensor_msgs::PointCloud2 cloud_msg;
// 	pcl::toROSMsg(*inPtr, cloud_msg);
// 	cloud_msg.header = header;
// 	in_publisher->publish(cloud_msg);
// }

// // void pubDetectedObject(const std::vector<Bndbox>& detections, const std_msgs::Header& in_header,autoware_msgs::DetectedObjectArray& objects)
// // {
// //   // std::cout << "detections :"<<detections.size()<<std::endl;
// //   objects.header = in_header;
// //   for (size_t i = 0; i < detections.size(); i++)
// //   {
// //     autoware_msgs::DetectedObject object;
// //     object.header = in_header;
// //     object.valid = true;
// //     object.pose_reliable = true;
    
// //     object.dimensions.x = detections[i].l;
// //     object.dimensions.y = detections[i].w;
// //     object.dimensions.z = detections[i].h;
// //     object.pose.position.x = detections[i].x + object.dimensions.x/2;
// //     object.pose.position.y = detections[i].y + object.dimensions.y/2;
// //     object.pose.position.z = detections[i].z + object.dimensions.z/2;

// //     // Trained this way
// //     float yaw = detections[i].rt;
// //     yaw += M_PI/2;
// //     yaw = std::atan2(std::sin(yaw), std::cos(yaw));
// //     geometry_msgs::Quaternion q = tf::createQuaternionMsgFromYaw(-yaw);
// //     object.pose.orientation = q;

// //     object.label = "car";
// //     objects.objects.push_back(object);
// //   }
// //   pub_objects_.publish(objects);
// // }


// void pubDetectedObject(const std::vector<Bndbox>& detections, const std_msgs::Header& in_header,autoware_msgs::DetectedObjectArray& objects)
// {
//   objects.header = in_header;
//   int num_objects = detections.size();
//   for (size_t i = 0; i < num_objects; i++)
//   {
//     autoware_msgs::DetectedObject object;
//     object.header = in_header;
//     object.valid = true;
//     object.pose_reliable = true;

//     object.dimensions.x = detections[i].l;
//     object.dimensions.y = detections[i].w;
//     object.dimensions.z = detections[i].h;
//     // object.pose.position.x = detections[i].x + object.dimensions.x/2;
//     // object.pose.position.y = detections[i].y + object.dimensions.y/2;
//     // object.pose.position.z = detections[i].z + object.dimensions.z/2;

//     object.pose.position.x = detections[i].x;
//     object.pose.position.y = detections[i].y;
//     object.pose.position.z = detections[i].z;
//     // object.pose.position.x = detections[i * OUTPUT_NUM_BOX_FEATURE_ + 0];
//     // object.pose.position.y = detections[i * OUTPUT_NUM_BOX_FEATURE_ + 1];
//     // object.pose.position.z = detections[i * OUTPUT_NUM_BOX_FEATURE_ + 2];

//     // Trained this way
//     float yaw = detections[i].rt;
//     yaw += M_PI/2;
//     yaw = std::atan2(std::sin(yaw), std::cos(yaw));
//     geometry_msgs::Quaternion q = tf::createQuaternionMsgFromYaw(-yaw);
//     object.pose.orientation = q;

//     // Again: Trained this way
//     object.dimensions.x = detections[i].l;
//     object.dimensions.y = detections[i].w;
//     object.dimensions.z = detections[i].h;
//     // object.dimensions.x = detections[i * OUTPUT_NUM_BOX_FEATURE_ + 4];
//     // object.dimensions.y = detections[i * OUTPUT_NUM_BOX_FEATURE_ + 3];
//     // object.dimensions.z = detections[i * OUTPUT_NUM_BOX_FEATURE_ + 5];

//     //Only detects car in Version 1.0
//     object.label = "car";
//     objects.objects.push_back(object);
//   }
//   pub_objects_.publish(objects);
// }
  


// int fileNameFilter(const struct dirent *cur) {
//     std::string str(cur->d_name);
//     if (str.find(".bin") != std::string::npos) {
//         return 1;
//     }
//     return 0;
// }
 
// std::vector<std::string> getDirBinsSortedPath(std::string dirPath) {
//     struct dirent **namelist;
//     std::vector<std::string> ret;
//     int n = scandir(dirPath.c_str(), &namelist, fileNameFilter, alphasort);
//     if (n < 0) {
//         return ret;
//     }
//     for (int i = 0; i < n; ++i) {
//         std::string filePath(namelist[i]->d_name);
//         ret.push_back(filePath);
//         free(namelist[i]);
//     };
//     free(namelist);
//     return ret;
// }

// int main(int argc, char **argv)
// {
//   ros::init(argc, argv, "lidar_point_pillars");
//   ros::NodeHandle nh_;
//   pub_in_cloud = nh_.advertise<sensor_msgs::PointCloud2>("/kitti/in_points", 1);
//   pub_objects_ = nh_.advertise<autoware_msgs::DetectedObjectArray>("/kitti/detection/lidar_detector/objects", 1);
//   publisher_markers_ =  nh_.advertise<visualization_msgs::MarkerArray>("/kitti/detection/visualize/cluster_markers", 1);

//   velodyneHeader.frame_id = "velodyne";

//   Getinfo();

//   cudaEvent_t start, stop;
//   float elapsedTime = 0.0f;
//   cudaStream_t stream = NULL;

//   checkCudaErrors(cudaEventCreate(&start));
//   checkCudaErrors(cudaEventCreate(&stop));
//   checkCudaErrors(cudaStreamCreate(&stream));

//   Params params_;

//   std::vector<Bndbox> nms_pred;
//   nms_pred.reserve(100);

//   PointPillar pointpillar(Model_File, stream);
//   std::vector<std::string> nameList = getDirBinsSortedPath(Data_File);
//   for (int i = 0; i < nameList.size(); i++)
//   {
//     std::string dataFile = Data_File;
//     dataFile += nameList[i];//包含后缀名.bin

//     std::cout << "<<<<<<<<<<<" <<std::endl;
//     std::cout << "load file: "<< dataFile <<std::endl;
    
//     //load points cloud
//     unsigned int length = 0;
//     void *data = NULL;
//     std::shared_ptr<char> buffer((char *)data, std::default_delete<char[]>());
//     pcl::PointCloud<pcl::PointXYZI>::Ptr cloudPtr(new pcl::PointCloud<pcl::PointXYZI>);
//     loadData(dataFile.data(), &data, &length,cloudPtr);
//     buffer.reset((char *)data);

//     float* points = (float*)buffer.get();
//     size_t points_size = length/sizeof(float)/4;

//     std::cout << "find points num: "<< points_size <<std::endl;

//     float *points_data = nullptr;
//     unsigned int points_data_size = points_size * 4 * sizeof(float);
//     // cudaMallocManaged 函数分配的内存，所有设备与主机均可访问
//     checkCudaErrors(cudaMallocManaged((void **)&points_data, points_data_size));
//     // 将主机端数据拷贝到GPU上
//     // 使用cudammcpy将数据复制到另一个缓冲区
//     checkCudaErrors(cudaMemcpy(points_data, points, points_data_size, cudaMemcpyDefault));
//     // 核函数调用之后必须调用cudaDeviceSynchronize 等待设备GPU完成访问，CPU才能进行访问
//     checkCudaErrors(cudaDeviceSynchronize());

//     cudaEventRecord(start, stream);

//     pointpillar.doinfer(points_data, points_size, nms_pred);
//     // cudaEventRecord是异步的，加入时间完全不可控，不能保证两个事件之间的间隔刚好是两个事件之间的
//     cudaEventRecord(stop, stream);
//     // 会阻塞主机线程知道事件被完成
//     cudaEventSynchronize(stop);
//     // 记录两个事件之间的时间间隔
//     cudaEventElapsedTime(&elapsedTime, start, stop);
//     std::cout<<"TIME: pointpillar: "<< elapsedTime <<" ms." <<std::endl;

//     checkCudaErrors(cudaFree(points_data));

//     std::cout<<"Bndbox objs: "<< nms_pred.size()<<std::endl;
//     std::cout << ">>>>>>>>>>>" <<std::endl;

    
//     publishCloud(&pub_in_cloud,velodyneHeader,cloudPtr);
//     autoware_msgs::DetectedObjectArray detected_objects;
//     pubDetectedObject(nms_pred,velodyneHeader,detected_objects);

//     // 可视化
//     visualization_msgs::MarkerArray visualize_markers;
//     vdo.visualizeDetectedObjs(detected_objects,visualize_markers);
//     publisher_markers_.publish(visualize_markers);
    
//     std::cout << "detected_objects :" <<detected_objects.objects.size()<<std::endl;
//     nms_pred.clear();
//   }

//   checkCudaErrors(cudaEventDestroy(start));
//   checkCudaErrors(cudaEventDestroy(stop));
//   checkCudaErrors(cudaStreamDestroy(stream));

//   return 0;
// }


