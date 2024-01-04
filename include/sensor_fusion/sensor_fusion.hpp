#ifndef SENSOR_FUSION_HPP_
#define SENSOR_FUSION_HPP_
#include "tier4_autoware_utils/tier4_autoware_utils.hpp"

#include <image_transport/image_transport.hpp>
#include <image_transport/subscriber_filter.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <message_filters/time_synchronizer.h>


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>

#include <nav_msgs/msg/odometry.hpp>
#include <radar_msgs/msg/radar_scan.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>
#include <tier4_perception_msgs/msg/detected_objects_with_feature.hpp>
#include <itri_msgs/msg/bboxs.hpp>
#include <cv_bridge/cv_bridge.h>


#include "autoware_auto_perception_msgs/msg/detected_objects.hpp"
#include <autoware_auto_perception_msgs/msg/predicted_objects.hpp>

#include <chrono>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace object_recognition
{
using autoware_auto_perception_msgs::msg::DetectedObject;
using autoware_auto_perception_msgs::msg::DetectedObjects;
using nav_msgs::msg::Odometry;
using radar_msgs::msg::RadarReturn;
using radar_msgs::msg::RadarScan;
using autoware_auto_perception_msgs::msg::PredictedObject;
using autoware_auto_perception_msgs::msg::PredictedObjects;
class SensorFusionNodelet : public rclcpp::Node
{
public:
  explicit SensorFusionNodelet(const rclcpp::NodeOptions & options);
  ~SensorFusionNodelet();

  struct NodeParam
  {
    double doppler_velocity_sd{};
  };
  float velocity;

  void callback_mode0(const tier4_perception_msgs::msg::DetectedObjectsWithFeature::ConstSharedPtr in_roi_msg,
                 const tier4_perception_msgs::msg::DetectedObjectsWithFeature::ConstSharedPtr in_roi_msg1,
                 const tier4_perception_msgs::msg::DetectedObjectsWithFeature::ConstSharedPtr in_roi_msg2,
                 const tier4_perception_msgs::msg::DetectedObjectsWithFeature::ConstSharedPtr in_roi_msg3, 
                 const RadarScan::ConstSharedPtr radar_msg1,
                 const RadarScan::ConstSharedPtr radar_msg2,
                 const RadarScan::ConstSharedPtr radar_msg3, 
                 const Odometry::ConstSharedPtr odom_msg);


  void callback_mode1(const tier4_perception_msgs::msg::DetectedObjectsWithFeature::ConstSharedPtr in_roi_msg,
                const tier4_perception_msgs::msg::DetectedObjectsWithFeature::ConstSharedPtr in_roi_msg1,
                const tier4_perception_msgs::msg::DetectedObjectsWithFeature::ConstSharedPtr in_roi_msg2,
                const tier4_perception_msgs::msg::DetectedObjectsWithFeature::ConstSharedPtr in_roi_msg3, 
                const DetectedObjects::ConstSharedPtr object_msg,
                const RadarScan::ConstSharedPtr radar_msg1,
                const RadarScan::ConstSharedPtr radar_msg2,
                const RadarScan::ConstSharedPtr radar_msg3, 
                const Odometry::ConstSharedPtr odom_msg);
private:
  // Subscriber
  message_filters::Subscriber<tier4_perception_msgs::msg::DetectedObjectsWithFeature> image_subs0_{};
  message_filters::Subscriber<tier4_perception_msgs::msg::DetectedObjectsWithFeature> image_subs1_{};
  message_filters::Subscriber<tier4_perception_msgs::msg::DetectedObjectsWithFeature> image_subs2_{};
  message_filters::Subscriber<tier4_perception_msgs::msg::DetectedObjectsWithFeature> image_subs3_{};
  message_filters::Subscriber<DetectedObjects> sub_object_{};
  message_filters::Subscriber<RadarScan> sub_radar_front_{};
  message_filters::Subscriber<RadarScan> sub_radar_left_rear_{};
  message_filters::Subscriber<RadarScan> sub_radar_right_rear_{};
  message_filters::Subscriber<Odometry> sub_odometry_{};
  std::shared_ptr<tier4_autoware_utils::TransformListener> transform_listener_;
  // 新增一個geometry_msgs::msg::Point類型的發布者
  rclcpp::Publisher<PredictedObjects>::SharedPtr pub_objects_;

  using SyncPolicy = message_filters::sync_policies::ApproximateTime<RadarScan, Odometry>;
  using Sync = message_filters::Synchronizer<SyncPolicy>;

  std::mutex connect_mutex_;


  typedef message_filters::sync_policies::ApproximateTime<
    tier4_perception_msgs::msg::DetectedObjectsWithFeature, 
    tier4_perception_msgs::msg::DetectedObjectsWithFeature,
    tier4_perception_msgs::msg::DetectedObjectsWithFeature, 
    tier4_perception_msgs::msg::DetectedObjectsWithFeature, 
    RadarScan, RadarScan, RadarScan, Odometry> ApproximateSyncPolicy_mode0;
  typedef message_filters::Synchronizer<ApproximateSyncPolicy_mode0> ApproximateSync_mode0;
  std::shared_ptr<ApproximateSync_mode0> approximate_sync_mode0;


    typedef message_filters::sync_policies::ApproximateTime<
    tier4_perception_msgs::msg::DetectedObjectsWithFeature, 
    tier4_perception_msgs::msg::DetectedObjectsWithFeature,
    tier4_perception_msgs::msg::DetectedObjectsWithFeature, 
    tier4_perception_msgs::msg::DetectedObjectsWithFeature, 
    DetectedObjects, RadarScan, RadarScan, RadarScan, Odometry> ApproximateSyncPolicy_mode1;
  typedef message_filters::Synchronizer<ApproximateSyncPolicy_mode1> ApproximateSync_mode1;
  std::shared_ptr<ApproximateSync_mode1> approximate_sync_mode1;

  rclcpp::TimerBase::SharedPtr timer_;
  

  // Parameter Server
  OnSetParametersCallbackHandle::SharedPtr set_param_res_;
  rcl_interfaces::msg::SetParametersResult onSetParam(
    const std::vector<rclcpp::Parameter> & params);

  // Parameter
  NodeParam node_param_{};

  // Function
  bool isStaticPointcloud(
    const RadarReturn & radar_return, const Odometry::ConstSharedPtr & odom_msg,
    geometry_msgs::msg::TransformStamped::ConstSharedPtr transform);
    
  bool isDataReady();

  // Data Buffer
  DetectedObjects::ConstSharedPtr detected_objects_{};

};

}  // namespace object_recognition

#endif  // SENSOR_FUSION__NODELET_HPP_
