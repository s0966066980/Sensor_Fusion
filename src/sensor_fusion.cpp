#include "sensor_fusion/sensor_fusion.hpp"

#include <autoware_auto_perception_msgs/msg/object_classification.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/twist_with_covariance.hpp>
#include <geometry_msgs/msg/vector3_stamped.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2/LinearMath/Transform.h>
#include <glob.h>
#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <filesystem>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;

// 用於排序的比較函數
bool compareFilesByTimestamp(const fs::path &a, const fs::path &b) {
    std::string nameA = a.stem().string();
    std::string nameB = b.stem().string();
    return std::stod(nameA) < std::stod(nameB);
}

namespace
{
template <class T>
bool update_param(
  const std::vector<rclcpp::Parameter> & params, const std::string & name, T & value)
{
  const auto itr = std::find_if(
    params.cbegin(), params.cend(),
    [&name](const rclcpp::Parameter & p) { return p.get_name() == name; });
  // Not found
  if (itr == params.cend()) {
    return false;
  }
  value = itr->template get_value<T>();
  return true;
}

geometry_msgs::msg::Vector3 getVelocity(const radar_msgs::msg::RadarReturn & radar)
{
  return geometry_msgs::build<geometry_msgs::msg::Vector3>()
    .x(radar.doppler_velocity * std::cos(radar.azimuth))
    .y(radar.doppler_velocity * std::sin(radar.azimuth))
    .z(0.0);
}

geometry_msgs::msg::Vector3 getTransformedVelocity(
  const geometry_msgs::msg::Vector3 velocity,
  geometry_msgs::msg::TransformStamped::ConstSharedPtr transform)
{
  geometry_msgs::msg::Vector3Stamped velocity_stamped{};
  velocity_stamped.vector = velocity;
  geometry_msgs::msg::Vector3Stamped transformed_velocity_stamped{};
  tf2::doTransform(velocity_stamped, transformed_velocity_stamped, *transform);
  return velocity_stamped.vector;
}

geometry_msgs::msg::Vector3 compensateEgoVehicleTwist(
  const radar_msgs::msg::RadarReturn & radar,
  const geometry_msgs::msg::TwistWithCovariance & ego_vehicle_twist_with_covariance,
  geometry_msgs::msg::TransformStamped::ConstSharedPtr transform)
{
  const geometry_msgs::msg::Vector3 radar_velocity = getVelocity(radar);
  const geometry_msgs::msg::Vector3 v_r = getTransformedVelocity(radar_velocity, transform);

  const auto v_e = ego_vehicle_twist_with_covariance.twist.linear;
  return geometry_msgs::build<geometry_msgs::msg::Vector3>()
    .x(v_r.x + v_e.x)
    .y(v_r.y + v_e.y)
    .z(v_r.z + v_e.z);
}
}  // namespace

namespace
{
std::vector<std::string> getFilePath(const std::string & input_dir)
{
  glob_t globbuf;
  std::vector<std::string> files;
  glob((input_dir + "*").c_str(), 0, NULL, &globbuf);
  for (size_t i = 0; i < globbuf.gl_pathc; i++) {
    files.push_back(globbuf.gl_pathv[i]);
  }
  globfree(&globbuf);
  return files;
}


struct RadarPoint {
  tf2::Vector3 point;
};

struct LabeledBoxPoint {
  tf2::Vector3 point;
  int label; // 假設標籤是整數類型，根據你的消息定義進行調整
};

struct LidarDetection {
  autoware_auto_perception_msgs::msg::PredictedObject object;
};

struct ClassifiedPoint {
    cv::Point2f point;
    int classId;
};


void processRadarPoint(const cv::Point2f& point, std::vector<cv::Point2f>& radarPoints) {
    radarPoints.push_back(point);
}
///radar_avg
cv::Point2f calculateAverage(const std::vector<cv::Point2f>& radarPoints) {
    float sum_x = 0, sum_y = 0;
    for (const auto& point : radarPoints) {
        sum_x += point.x;
        sum_y += point.y;
    }
    return cv::Point2f(sum_x / radarPoints.size(), sum_y / radarPoints.size());
}

bool isPointInRadarRange(const cv::Point2f& point, const std::vector<cv::Point2f>& averageRadarPoints, float range) {
    int check = 0;
    for (const auto& avgRadarPoint : averageRadarPoints) {
        if (std::abs(point.x - avgRadarPoint.x) < range){
            check = 1;
        }
    }
    if(check == 1){
        return false;
    }
    else
        return true;
}

std::vector<tf2::Vector3> box_points;
std::vector<std::vector<LabeledBoxPoint>> all_labeled_box_points;
std::vector<LidarDetection> all_lidar_detection;
std::vector<std::vector<RadarPoint>> all_radar_points_F;
std::vector<std::vector<RadarPoint>> all_radar_points_L;
std::vector<std::vector<RadarPoint>> all_radar_points_R;
std::vector<ClassifiedPoint> classifiedPoints;
std::vector<cv::Point2f> transformedPoints;
std::vector<cv::Point2f> points2f;
std::vector<cv::Point2f> average_radar_points;
std::set<int> lidar_labels;
cv::Mat FrontIPM;
cv::Mat LeftIPM;
cv::Mat RightIPM;
cv::Mat RearIPM;
int MODEL_WIDTH= 416, MODEL_HEIGHT= 256;
cv::Point2f Front[] =
{
  { static_cast<float>(MODEL_WIDTH) *static_cast<float>(0.413398),  static_cast<float>(MODEL_HEIGHT) *static_cast<float>(0.545477)},
  { static_cast<float>(MODEL_WIDTH) *static_cast<float>(0.586603),  static_cast<float>(MODEL_HEIGHT) *static_cast<float>(0.545477)},
  { static_cast<float>(MODEL_WIDTH) *static_cast<float>(0.066990),  static_cast<float>(MODEL_HEIGHT) *static_cast<float>(0.727387)},
  { static_cast<float>(MODEL_WIDTH) *static_cast<float>(0.933014),  static_cast<float>(MODEL_HEIGHT) *static_cast<float>(0.727387)}
};
cv::Point2f Right[] =
{
  { static_cast<float>(MODEL_WIDTH) *static_cast<float>(0.409007),  static_cast<float>(MODEL_HEIGHT) *static_cast<float>(0.526878)},
  { static_cast<float>(MODEL_WIDTH) *static_cast<float>(0.590993),  static_cast<float>(MODEL_HEIGHT) *static_cast<float>(0.526878)},
  { static_cast<float>(MODEL_WIDTH) *static_cast<float>(0.045027),  static_cast<float>(MODEL_HEIGHT) *static_cast<float>(0.634392)},
  { static_cast<float>(MODEL_WIDTH) *static_cast<float>(0.954970),  static_cast<float>(MODEL_HEIGHT) *static_cast<float>(0.634392)}
};
cv::Point2f Left[] =
{
  { static_cast<float>(MODEL_WIDTH) *static_cast<float>(0.409007),  static_cast<float>(MODEL_HEIGHT) *static_cast<float>(0.526878)},
  { static_cast<float>(MODEL_WIDTH) *static_cast<float>(0.590993),  static_cast<float>(MODEL_HEIGHT) *static_cast<float>(0.526878)},
  { static_cast<float>(MODEL_WIDTH) *static_cast<float>(0.045027),  static_cast<float>(MODEL_HEIGHT) *static_cast<float>(0.634392)},
  { static_cast<float>(MODEL_WIDTH) *static_cast<float>(0.954970),  static_cast<float>(MODEL_HEIGHT) *static_cast<float>(0.634392)}
};
cv::Point2f Rear[] =
{
  { static_cast<float>(MODEL_WIDTH) *static_cast<float>(0.409008),  static_cast<float>(MODEL_HEIGHT) *static_cast<float>(0.532851)},
  { static_cast<float>(MODEL_WIDTH) *static_cast<float>(0.590993),  static_cast<float>(MODEL_HEIGHT) *static_cast<float>(0.532851)},
  { static_cast<float>(MODEL_WIDTH) *static_cast<float>(0.045039),  static_cast<float>(MODEL_HEIGHT) *static_cast<float>(0.664253)},
  { static_cast<float>(MODEL_WIDTH) *static_cast<float>(0.954966),  static_cast<float>(MODEL_HEIGHT) *static_cast<float>(0.664253)}
};

//距離(m)
cv::Point2f dst60[] =
{
  { -5     ,   50 },
  {  5     ,   50 },
  { -5     ,   10 }, //10
  {  5     ,   10 }  //10
};

//距離(m)
cv::Point2f dst140[] =
{
  { -5     ,   10 },
  {  5     ,   10 },
  { -5     ,   2  },
  {  5     ,   2  }
};


}  // namespace
namespace object_recognition
{
SensorFusionNodelet::SensorFusionNodelet(const rclcpp::NodeOptions & options)
: Node("sensor_fusion", options)
{
    using std::placeholders::_1;
    using std::placeholders::_2;
    const int mode = static_cast<int>(this->declare_parameter<int>("mode"));
    image_subs0_.subscribe(this, "perception/object_recognition/detection/front_narrow/obstacle_objects", rmw_qos_profile_sensor_data);
    image_subs1_.subscribe(this, "perception/object_recognition/detection/left/obstacle_objects", rmw_qos_profile_sensor_data);
    image_subs2_.subscribe(this, "perception/object_recognition/detection/right/obstacle_objects", rmw_qos_profile_sensor_data);
    image_subs3_.subscribe(this, "perception/object_recognition/detection/rear/obstacle_objects", rmw_qos_profile_sensor_data);
    sub_object_.subscribe(this, "/output/objects", rmw_qos_profile_sensor_data);
    sub_radar_front_.subscribe(this, "~/input/radar_front", rmw_qos_profile_sensor_data);
    sub_radar_left_rear_.subscribe(this, "~/input/radar_left_rear", rmw_qos_profile_sensor_data);
    sub_radar_right_rear_.subscribe(this, "~/input/radar_right_rear", rmw_qos_profile_sensor_data);
    sub_odometry_.subscribe(this, "~/input/odometry", rmw_qos_profile_sensor_data);
    
    if(mode==0){
        approximate_sync_mode0.reset(new ApproximateSync_mode0(ApproximateSyncPolicy_mode0(50), image_subs0_, image_subs1_,image_subs2_, image_subs3_, 
                                                    sub_radar_front_, sub_radar_left_rear_, sub_radar_right_rear_, sub_odometry_));
        approximate_sync_mode0->registerCallback(
            std::bind(&SensorFusionNodelet::callback_mode0, this, std::placeholders::_1,
                    std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5, 
                    std::placeholders::_6, std::placeholders::_7, std::placeholders::_8));
    }

    if(mode==1){
        approximate_sync_mode1.reset(new ApproximateSync_mode1(ApproximateSyncPolicy_mode1(50), image_subs0_, image_subs1_,image_subs2_, image_subs3_, sub_object_, 
                                                sub_radar_front_, sub_radar_left_rear_, sub_radar_right_rear_, sub_odometry_));
        approximate_sync_mode1->registerCallback(
            std::bind(&SensorFusionNodelet::callback_mode1, this, std::placeholders::_1,
                    std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5, 
                    std::placeholders::_6, std::placeholders::_7, std::placeholders::_8, std::placeholders::_9));
    }

    // Parameter Server
    set_param_res_ = this->add_on_set_parameters_callback(
        std::bind(&SensorFusionNodelet::onSetParam, this, std::placeholders::_1));


    // Node Parameter
    node_param_.doppler_velocity_sd = declare_parameter<double>("doppler_velocity_sd", 2.0);

    // Subscriber
    transform_listener_ = std::make_shared<tier4_autoware_utils::TransformListener>(this);
    pub_objects_ = this->create_publisher<PredictedObjects>("fusion_point", 10);
}

rcl_interfaces::msg::SetParametersResult SensorFusionNodelet::onSetParam(
    const std::vector<rclcpp::Parameter> & params)
{
    rcl_interfaces::msg::SetParametersResult result;
    try {
        auto & p = node_param_;
        update_param(params, "doppler_velocity_sd", p.doppler_velocity_sd);
    } catch (const rclcpp::exceptions::InvalidParameterTypeException & e) {
        result.successful = false;
        result.reason = e.what();
        return result;
    }
    result.successful = true;
    result.reason = "success";
    return result;
}

void SensorFusionNodelet::callback_mode0(const tier4_perception_msgs::msg::DetectedObjectsWithFeature::ConstSharedPtr in_roi_msg,
                                    const tier4_perception_msgs::msg::DetectedObjectsWithFeature::ConstSharedPtr in_roi_msg1,
                                    const tier4_perception_msgs::msg::DetectedObjectsWithFeature::ConstSharedPtr in_roi_msg2,
                                    const tier4_perception_msgs::msg::DetectedObjectsWithFeature::ConstSharedPtr in_roi_msg3,
                                    const RadarScan::ConstSharedPtr radar_msg1,
                                    const RadarScan::ConstSharedPtr radar_msg2,
                                    const RadarScan::ConstSharedPtr radar_msg3, 
                                    const Odometry::ConstSharedPtr odom_msg)
{
    
    all_radar_points_F.clear();
    all_radar_points_L.clear();
    all_radar_points_R.clear();

    if (!radar_msg1->returns.empty()) {
        geometry_msgs::msg::TransformStamped::ConstSharedPtr transform;

        velocity = odom_msg->twist.twist.linear.x;
        try {
            transform = transform_listener_->getTransform(
            odom_msg->header.frame_id, radar_msg1->header.frame_id, odom_msg->header.stamp,
            rclcpp::Duration::from_seconds(0.2));
        } catch (tf2::TransformException & ex) {
            RCLCPP_INFO(this->get_logger(), "Could not transform");
            return;
        }

        for (const auto & radar_return1 : radar_msg1->returns) {
            if (!isStaticPointcloud(radar_return1, odom_msg, transform)) {
                float Y = radar_return1.range * sin(radar_return1.azimuth);
                float X = radar_return1.range * cos(radar_return1.azimuth) + 2.3;
                float Z = radar_return1.range * sin(radar_return1.elevation) - 1.6;

                // 創建tf2::Vector3來存儲雷達點
                tf2::Vector3 radar_point(Y, X, Z);

                // 計算旋轉
                tf2::Quaternion q;
                q.setRPY(0, 0, M_PI * 1.5); // 1.5 * π 是 270 度

                // 進行座標轉換
                tf2::Transform transform(q);
                tf2::Vector3 transformed_point = transform * radar_point;

                std::vector<RadarPoint> radar_pointsF = {{transformed_point}};
                all_radar_points_F.push_back(radar_pointsF);
            }
        }
    }
    else {
        RCLCPP_WARN(this->get_logger(), "No radar detected in object_msg");
    }

    if (!radar_msg2->returns.empty()) {
        geometry_msgs::msg::TransformStamped::ConstSharedPtr transform;

        velocity = odom_msg->twist.twist.linear.x;
        nav_msgs::msg::Odometry::SharedPtr newOdometry = std::make_shared<nav_msgs::msg::Odometry>();
        *newOdometry = *odom_msg;
        newOdometry->twist.twist.linear.x = odom_msg->twist.twist.linear.x * -1;//轉180度
    
        try {
            transform = transform_listener_->getTransform(
            newOdometry->header.frame_id, radar_msg2->header.frame_id, newOdometry->header.stamp,
            rclcpp::Duration::from_seconds(0.2));
        } catch (tf2::TransformException & ex) {
            RCLCPP_INFO(this->get_logger(), "Could not transform");
            return;
        }

        for (const auto & radar_return2 : radar_msg2->returns) {
            if (!isStaticPointcloud(radar_return2, newOdometry, transform))
            {
                float Y = radar_return2.range * sin(radar_return2.azimuth) + 0.8;
                float X = radar_return2.range * cos(radar_return2.azimuth) + 2.3;
                float Z = radar_return2.range * sin(radar_return2.elevation) - 1.6;

                // 創建tf2::Vector3來存儲雷達點
                tf2::Vector3 radar_point(Y, X, Z);

                // 計算旋轉
                tf2::Quaternion q;
                q.setRPY(0, 0, M_PI * 0.5); // 1.5 * π 是 270 度

                // 進行座標轉換
                tf2::Transform transform(q);
                tf2::Vector3 transformed_point = transform * radar_point;

                std::vector<RadarPoint> radar_pointsL = {{transformed_point}};
                all_radar_points_L.push_back(radar_pointsL);
            }
        }
    }
    else {
        RCLCPP_WARN(this->get_logger(), "No radar detected in object_msg");
    }

    if (!radar_msg3->returns.empty()) {
        geometry_msgs::msg::TransformStamped::ConstSharedPtr transform;

        velocity = odom_msg->twist.twist.linear.x;
        nav_msgs::msg::Odometry::SharedPtr newOdometry = std::make_shared<nav_msgs::msg::Odometry>();
        *newOdometry = *odom_msg;
        newOdometry->twist.twist.linear.x = odom_msg->twist.twist.linear.x * -1;//轉180度
        
        try {
            transform = transform_listener_->getTransform(
            newOdometry->header.frame_id, radar_msg3->header.frame_id, newOdometry->header.stamp,
            rclcpp::Duration::from_seconds(0.2));
        } catch (tf2::TransformException & ex) {
            RCLCPP_INFO(this->get_logger(), "Could not transform");
            return;
        }

        for (const auto & radar_return3 : radar_msg3->returns) {
            if (!isStaticPointcloud(radar_return3, newOdometry, transform))
            {
                float Y = radar_return3.range * sin(radar_return3.azimuth) - 0.8;
                float X = radar_return3.range * cos(radar_return3.azimuth) + 2.3;
                float Z = radar_return3.range * sin(radar_return3.elevation + 3.2);

                // 創建tf2::Vector3來存儲雷達點
                tf2::Vector3 radar_point(Y, X, Z);

                // 計算旋轉
                tf2::Quaternion q;
                q.setRPY(0, 0, M_PI * 0.5); // 1.5 * π 是 270 度

                // 進行座標轉換
                tf2::Transform transform(q);
                tf2::Vector3 transformed_point = transform * radar_point;

                std::vector<RadarPoint> radar_pointsR = {{transformed_point}};
                all_radar_points_R.push_back(radar_pointsR);
            }
        }
    }
    else {
        RCLCPP_WARN(this->get_logger(), "No radar detected in object_msg");
    }
    
    // RCLCPP_INFO(this->get_logger(), "Timestamp for image 1: %d.%09d", in_roi_msg->header.stamp.sec, in_roi_msg->header.stamp.nanosec);
    std::array<tier4_perception_msgs::msg::DetectedObjectsWithFeature::ConstSharedPtr, 4> in_roi_msgs = {in_roi_msg, in_roi_msg1, in_roi_msg2, in_roi_msg3};

    
    FrontIPM = cv::getPerspectiveTransform(Front, dst60);
    LeftIPM = cv::getPerspectiveTransform(Left, dst140);
    RightIPM = cv::getPerspectiveTransform(Right, dst140);
    RearIPM = cv::getPerspectiveTransform(Rear, dst140);

    for (int i = 0; i < in_roi_msgs.size(); i++)
    {
        for (const auto &object : in_roi_msgs[i]->feature_objects) {
            points2f.clear();
            if(i==0){
                if(object.object.classification.front().label>0 && object.object.classification.front().label <=7){

                    points2f.push_back(cv::Point2f(object.feature.roi.x_offset, object.feature.roi.y_offset + object.feature.roi.height)); // 左下角
                    points2f.push_back(cv::Point2f(object.feature.roi.x_offset + object.feature.roi.width, object.feature.roi.y_offset + object.feature.roi.height)); // 右下角
                    // points2f.push_back(cv::Point2f(object.feature.roi.x_offset, object.feature.roi.y_offset + object.feature.roi.height)); // 左下角
                    // points2f.push_back(cv::Point2f(object.feature.roi.x_offset + object.feature.roi.width, object.feature.roi.y_offset + object.feature.roi.height)); // 右下角
                }
                
                if(points2f.size() > 0)
                {
                    transformedPoints.clear();
                    cv::perspectiveTransform(points2f, transformedPoints, FrontIPM);

                    for (size_t j = 0; j < transformedPoints.size(); ++j) {
                        // std::cout << transformedPoints[j].x << "  "  << transformedPoints[j].y <<std::endl;
                        // std::cout << "-----------------------" <<std::endl;
                        transformedPoints[j].y =  transformedPoints[j].y + 0.7;
                        classifiedPoints.push_back({transformedPoints[j], object.object.classification.front().label});
                    }
                }
            }
            if(i==1){
                if(object.object.classification.front().label>=0 && object.object.classification.front().label <=7){

                    points2f.push_back(cv::Point2f(object.feature.roi.x_offset, object.feature.roi.y_offset + object.feature.roi.height)); // 左下角
                    points2f.push_back(cv::Point2f(object.feature.roi.x_offset + object.feature.roi.width, object.feature.roi.y_offset + object.feature.roi.height)); // 右下角
                    // points2f.push_back(cv::Point2f(object.feature.roi.x_offset, object.feature.roi.y_offset + object.feature.roi.height)); // 左下角
                    // points2f.push_back(cv::Point2f(object.feature.roi.x_offset + object.feature.roi.width, object.feature.roi.y_offset + object.feature.roi.height)); // 右下角
                }
                
                if(points2f.size() > 0)
                {
                    transformedPoints.clear();
                    cv::perspectiveTransform(points2f, transformedPoints, LeftIPM);

                    for (size_t j = 0; j < transformedPoints.size(); ++j) {
                        // std::cout << -transformedPoints[j].y << "  "  << transformedPoints[j].x <<std::endl;
                        // std::cout << "-----------------------" <<std::endl;
                        float new_x = transformedPoints[j].y;
                        float new_y = -transformedPoints[j].x;
                        transformedPoints[j].x = new_x - 1;
                        transformedPoints[j].y = new_y + 0.3;
                        classifiedPoints.push_back({transformedPoints[j], object.object.classification.front().label});
                    }
                }
            }
            if(i==2){
                if(object.object.classification.front().label>0 && object.object.classification.front().label <=7){

                    points2f.push_back(cv::Point2f(object.feature.roi.x_offset, object.feature.roi.y_offset + object.feature.roi.height)); // 左下角
                    points2f.push_back(cv::Point2f(object.feature.roi.x_offset + object.feature.roi.width, object.feature.roi.y_offset + object.feature.roi.height)); // 右下角
                    // points2f.push_back(cv::Point2f(object.feature.roi.x_offset, object.feature.roi.y_offset + object.feature.roi.height)); // 左下角
                    // points2f.push_back(cv::Point2f(object.feature.roi.x_offset + object.feature.roi.width, object.feature.roi.y_offset + object.feature.roi.height)); // 右下角
                }
                
                if(points2f.size() > 0)
                {
                    transformedPoints.clear();
                    
                    cv::perspectiveTransform(points2f, transformedPoints, RightIPM);
                    for (size_t j = 0; j < transformedPoints.size(); ++j) {
                        // std::cout << -transformedPoints[j].y << "  "  << transformedPoints[j].x <<std::endl;
                        // std::cout << "-----------------------" <<std::endl;
                        float new_x = transformedPoints[j].y;
                        float new_y = -transformedPoints[j].x;
                        transformedPoints[j].x = new_x + 1;
                        transformedPoints[j].y = new_y + 0.3;
                        classifiedPoints.push_back({transformedPoints[j], object.object.classification.front().label});
                    }
                }
            }
            if(i==3){
                if(object.object.classification.front().label>0 && object.object.classification.front().label <=7){

                    points2f.push_back(cv::Point2f(object.feature.roi.x_offset, object.feature.roi.y_offset + object.feature.roi.height)); // 左下角
                    points2f.push_back(cv::Point2f(object.feature.roi.x_offset + object.feature.roi.width, object.feature.roi.y_offset + object.feature.roi.height)); // 右下角
                    // points2f.push_back(cv::Point2f(object.feature.roi.x_offset, object.feature.roi.y_offset + object.feature.roi.height)); // 左下角
                    // points2f.push_back(cv::Point2f(object.feature.roi.x_offset + object.feature.roi.width, object.feature.roi.y_offset + object.feature.roi.height)); // 右下角
                }
                
                if(points2f.size() > 0)
                {
                    transformedPoints.clear();
                    cv::perspectiveTransform(points2f, transformedPoints, RearIPM);

                    for (size_t j = 0; j < transformedPoints.size(); ++j) {
                        // std::cout << -transformedPoints[j].y << "  "  << transformedPoints[j].x <<std::endl;
                        // std::cout << "-----------------------" <<std::endl;
                        transformedPoints[j].y = (-1 * transformedPoints[j].y) - 2.3;
                        classifiedPoints.push_back({transformedPoints[j], object.object.classification.front().label});
                    }
                }
            }
        }
    }
    // RCLCPP_INFO(this->get_logger(), "Timestamp for image 1: %d.%09d", in_roi_msg->header.stamp.sec, in_roi_msg->header.stamp.nanosec);

    // 在Lidar_Detection_msg的if判斷之後，建立檔案名稱和開啟txt檔案
    std::string file_name = "/home/lewin/carla-autoware-universe/autoware/src/universe/autoware.universe/perception/sensor_fusion/result/" + std::to_string(in_roi_msg->header.stamp.sec) + "." + std::to_string(in_roi_msg->header.stamp.nanosec) + ".txt";
    std::ofstream outfile(file_name);
    // 假設有一個集合來存儲所有從LiDAR資料中發現的標籤
    lidar_labels.clear();
    PredictedObjects output;
    output.header = in_roi_msg->header;
    output.header.frame_id = "map";
    PredictedObject point_msg;
    PredictedObject min_point;
    PredictedObject lidar_objects;

    // std::cout<<"object_msg: " + std::to_string(object_msg->header.stamp.sec) + "." + std::to_string(object_msg->header.stamp.nanosec) << std::endl;
    // std::cout<<"in_roi_msg: " + std::to_string(in_roi_msg->header.stamp.sec) + "." + std::to_string(in_roi_msg->header.stamp.nanosec) << std::endl;

    if (outfile.is_open())
    {
        if(!all_radar_points_F.empty() || !all_radar_points_L.empty() || !all_radar_points_R.empty()){
            average_radar_points.clear();
            if(!all_radar_points_F.empty()){
                std::vector<cv::Point2f> temp_radar_points;
                bool is_first_point = true;
                for (const auto& radar_points : all_radar_points_F) {
                    for (const auto& radar_point : radar_points) {
                        outfile << "2 " << radar_point.point.x() << " " << radar_point.point.y() << " " << radar_point.point.z() << " " << -1 << std::endl;
                        processRadarPoint(cv::Point2f(radar_point.point.x(), radar_point.point.y()), temp_radar_points);
                        // 檢查是否為第一個點或是否有更小的x座標
                        if (is_first_point || radar_point.point.x() < min_point.kinematics.initial_pose_with_covariance.pose.position.x) {
                            min_point.kinematics.initial_pose_with_covariance.pose.position.x = radar_point.point.x();
                            min_point.kinematics.initial_pose_with_covariance.pose.position.y = radar_point.point.y();
                            min_point.kinematics.initial_pose_with_covariance.pose.position.z = radar_point.point.z();
                            is_first_point = false;
                        }
                    }
                }
                if (!temp_radar_points.empty()) {
                    cv::Point2f averagePoint = calculateAverage(temp_radar_points);
                    average_radar_points.push_back(averagePoint);
                }
                
                    // 只有當至少有一個點時才發布
                if (!is_first_point) {
                    output.objects.push_back(min_point);
                }
            }
            if(!all_radar_points_L.empty() && !all_radar_points_R.empty()){
                // 初始化一個變數來存儲x座標最小的點
                // PredictedObject min_point;
                // RCLCPP_INFO(this->get_logger(), "222");
                std::vector<cv::Point2f> temp_radar_points;
                bool is_first_point = true;
                for (const auto& radar_points : all_radar_points_L) {
                    for (const auto& radar_point : radar_points) {
                        outfile << "2 " << radar_point.point.x() << " " << radar_point.point.y() << " " << radar_point.point.z() << " " << -2 << std::endl;
                        processRadarPoint(cv::Point2f(radar_point.point.x(), radar_point.point.y()), temp_radar_points);                            // 檢查是否為第一個點或是否有更小的x座標
                        if (is_first_point || radar_point.point.x() < min_point.kinematics.initial_pose_with_covariance.pose.position.x) {
                            min_point.kinematics.initial_pose_with_covariance.pose.position.x = radar_point.point.x();
                            min_point.kinematics.initial_pose_with_covariance.pose.position.y = radar_point.point.y();
                            min_point.kinematics.initial_pose_with_covariance.pose.position.z = radar_point.point.z();
                            is_first_point = false;
                        }
                    }
                }
                for (const auto& radar_points : all_radar_points_R) {
                    for (const auto& radar_point : radar_points) {
                        outfile << "2 " << radar_point.point.x() << " " << radar_point.point.y() << " " << radar_point.point.z() << " " << -2 << std::endl;
                        processRadarPoint(cv::Point2f(radar_point.point.x(), radar_point.point.y()), temp_radar_points);                            // 檢查是否為第一個點或是否有更小的x座標
                        // 檢查是否為第一個點或是否有更小的x座標
                        if (is_first_point || radar_point.point.x() < min_point.kinematics.initial_pose_with_covariance.pose.position.x) {
                        min_point.kinematics.initial_pose_with_covariance.pose.position.x = radar_point.point.x();
                        min_point.kinematics.initial_pose_with_covariance.pose.position.y = radar_point.point.y();
                        min_point.kinematics.initial_pose_with_covariance.pose.position.z = radar_point.point.z();
                        is_first_point = false;
                        }
                    }
                }

                if (!temp_radar_points.empty()) {
                    cv::Point2f averagePoint = calculateAverage(temp_radar_points);
                    average_radar_points.push_back(averagePoint);
                }

                // 只有當至少有一個點時才發布
                if (!is_first_point) {
                    output.objects.push_back(min_point);
                }
            }
            else{
                if(!all_radar_points_L.empty()){
                    // 初始化一個變數來存儲x座標最小的點
                    // PredictedObject min_point;
                    // RCLCPP_INFO(this->get_logger(), "333");
                    std::vector<cv::Point2f> temp_radar_points;
                    bool is_first_point = true;
                    for (const auto& radar_points : all_radar_points_L) {
                        for (const auto& radar_point : radar_points) {
                            outfile << "2 " << radar_point.point.x() << " " << radar_point.point.y() << " " << radar_point.point.z() << " " << -2 << std::endl;
                            processRadarPoint(cv::Point2f(radar_point.point.x(), radar_point.point.y()), temp_radar_points);                            // 檢查是否為第一個點或是否有更小的x座標
                            // 檢查是否為第一個點或是否有更小的x座標
                            if (is_first_point || radar_point.point.x() < min_point.kinematics.initial_pose_with_covariance.pose.position.x) {
                            min_point.kinematics.initial_pose_with_covariance.pose.position.x = radar_point.point.x();
                            min_point.kinematics.initial_pose_with_covariance.pose.position.y = radar_point.point.y();
                            min_point.kinematics.initial_pose_with_covariance.pose.position.z = radar_point.point.z();
                            is_first_point = false;
                            }
                        }
                    }

                    if (!temp_radar_points.empty()) {
                        cv::Point2f averagePoint = calculateAverage(temp_radar_points);
                        average_radar_points.push_back(averagePoint);
                    }

                    // 只有當至少有一個點時才發布
                    if (!is_first_point) {
                        output.objects.push_back(min_point);
                    }
                }
                if(!all_radar_points_R.empty()){
                    // 初始化一個變數來存儲x座標最小的點
                    // PredictedObject min_point;
                    // RCLCPP_INFO(this->get_logger(), "333");
                    std::vector<cv::Point2f> temp_radar_points;
                    bool is_first_point = true;
                    for (const auto& radar_points : all_radar_points_R) {
                        for (const auto& radar_point : radar_points) {
                            outfile << "2 " << radar_point.point.x() << " " << radar_point.point.y() << " " << radar_point.point.z() << " " << -3 << std::endl;
                            processRadarPoint(cv::Point2f(radar_point.point.x(), radar_point.point.y()), temp_radar_points);                            // 檢查是否為第一個點或是否有更小的x座標
                            // 檢查是否為第一個點或是否有更小的x座標
                            if (is_first_point || radar_point.point.x() < min_point.kinematics.initial_pose_with_covariance.pose.position.x) {
                            min_point.kinematics.initial_pose_with_covariance.pose.position.x = radar_point.point.x();
                            min_point.kinematics.initial_pose_with_covariance.pose.position.y = radar_point.point.y();
                            min_point.kinematics.initial_pose_with_covariance.pose.position.z = radar_point.point.z();
                            is_first_point = false;
                            }
                        }
                    }

                    if (!temp_radar_points.empty()) {
                        cv::Point2f averagePoint = calculateAverage(temp_radar_points);
                        average_radar_points.push_back(averagePoint);
                    }

                    // 只有當至少有一個點時才發布
                    if (!is_first_point) {
                        output.objects.push_back(min_point);
                    }
                }
            }

            if(!classifiedPoints.empty()) {
                // 存取bounding box的資料
                // PredictedObject point_msg;
                int label = 0, check = 0;
                float new_x = 0, new_y = 0;
                float pre_x = 0, pre_y = 0;

                for (const auto& classifiedPoint : classifiedPoints) {
                    if(classifiedPoint.classId>0 && classifiedPoint.classId<=4){
                        label = 2;
                    }
                    else if(classifiedPoint.classId>=5 && classifiedPoint.classId<=6){
                        label = 1;
                    }
                    else if(classifiedPoint.classId==7){
                        label = 0;
                    }
                    
                    new_x += classifiedPoint.point.y;
                    new_y += -classifiedPoint.point.x;

                    if(check % 2 ==0){
                        pre_x = classifiedPoint.point.y;
                        pre_y = -classifiedPoint.point.x;
                    }

                    ++check;

                    if(check % 2 == 0){
                        cv::Point2f centerPoint(new_x / 2, new_y / 2);
                        std::cout << centerPoint << std::endl;
                        if(isPointInRadarRange(centerPoint, average_radar_points, 5.0)){
                            outfile << "0 " << pre_x << " " << pre_y << " " << "0" << " " << label << std::endl;
                            outfile << "0 " << classifiedPoint.point.y << " " << -classifiedPoint.point.x << " " << "0" << " " << label << std::endl;
                            point_msg.kinematics.initial_pose_with_covariance.pose.position.x = new_x/2;
                            point_msg.kinematics.initial_pose_with_covariance.pose.position.y = new_y/2;
                            // point_msg.kinematics.initial_pose_with_covariance.pose.orientation =
                            //   tier4_autoware_utils::createQuaternionFromYaw(0.0);
                            // point_msg.kinematics.initial_twist_with_covariance.twist.linear.x = 0.0;
                            point_msg.shape.dimensions.x = 1.0;
                            point_msg.shape.dimensions.y = 1.0;
                            output.objects.push_back(point_msg);
                            new_x = 0;
                            new_y = 0;
                            pre_x = 0;
                            pre_y = 0;
                        }
                    }
                }
            }
        }
        else{
            if(!classifiedPoints.empty()) {
                // 存取bounding box的資料
                // PredictedObject point_msg;
                int label = 0;
                float new_x = 0;
                float new_y = 0;
                int check = 0;
                for (const auto& classifiedPoint : classifiedPoints) {
                    if(classifiedPoint.classId>0 && classifiedPoint.classId<=4){
                        label = 2;
                    }
                    else if(classifiedPoint.classId>=5 && classifiedPoint.classId<=6){
                        label = 1;
                    }
                    else if(classifiedPoint.classId==7){
                        label = 0;
                    }
                    
                    outfile << "0 " << classifiedPoint.point.y << " " << -classifiedPoint.point.x << " " << "0" << " " << label << std::endl;

                    new_x += classifiedPoint.point.y;
                    new_y += -classifiedPoint.point.x;
                    check++;

                    if(check == 2 ){
                        point_msg.kinematics.initial_pose_with_covariance.pose.position.x = new_x/2;
                        point_msg.kinematics.initial_pose_with_covariance.pose.position.y = new_y/2;
                        // point_msg.kinematics.initial_pose_with_covariance.pose.orientation =
                        //   tier4_autoware_utils::createQuaternionFromYaw(0.0);
                        // point_msg.kinematics.initial_twist_with_covariance.twist.linear.x = 0.0;
                        point_msg.shape.dimensions.x = 1.0;
                        point_msg.shape.dimensions.y = 1.0;
                        output.objects.push_back(point_msg);
                        new_x = 0;
                        new_y = 0;
                        check = 0;
                    }

                }
            }
        }
        // Publish Results
        pub_objects_->publish(output);
        outfile.close();
    } 
    else 
    {
        RCLCPP_ERROR(this->get_logger(), "無法打開檔案: %s", file_name.c_str());
    }
    classifiedPoints.clear();

}


void SensorFusionNodelet::callback_mode1(const tier4_perception_msgs::msg::DetectedObjectsWithFeature::ConstSharedPtr in_roi_msg,
                                    const tier4_perception_msgs::msg::DetectedObjectsWithFeature::ConstSharedPtr in_roi_msg1,
                                    const tier4_perception_msgs::msg::DetectedObjectsWithFeature::ConstSharedPtr in_roi_msg2,
                                    const tier4_perception_msgs::msg::DetectedObjectsWithFeature::ConstSharedPtr in_roi_msg3,
                                    const DetectedObjects::ConstSharedPtr object_msg,
                                    const RadarScan::ConstSharedPtr radar_msg1,
                                    const RadarScan::ConstSharedPtr radar_msg2,
                                    const RadarScan::ConstSharedPtr radar_msg3, 
                                    const Odometry::ConstSharedPtr odom_msg)
{
    // RCLCPP_INFO(this->get_logger(), "Timestamp for image 1: %d.%09d", object_msg->header.stamp.sec, object_msg->header.stamp.nanosec);
    // Lidar_Detection_msg
    if (!object_msg->objects.empty()) {
        all_labeled_box_points.clear();
        all_lidar_detection.clear();
        for (const auto& object : object_msg->objects) {
            auto position = object.kinematics.pose_with_covariance.pose.position;
            auto orientation = object.kinematics.pose_with_covariance.pose.orientation;
            auto dimensions = object.shape.dimensions;
            auto twist_with_covariance = object.kinematics.twist_with_covariance.twist.linear;
            int label = object.classification.begin()->label; // 假設我們只關注第一個分類標籤
            int classes = 2;
            tf2::Quaternion quat(orientation.x, orientation.y, orientation.z, orientation.w);
            tf2::Transform transform(quat, tf2::Vector3(position.x, position.y, position.z));
            if(label == 5 || label == 6) {
            classes = 1;
            } else if(label == 7) {
            classes = 0;
            }
            std::vector<LabeledBoxPoint> labeled_box_points = {
            {transform * tf2::Vector3(-dimensions.x/2, -dimensions.y/2, dimensions.z/2), classes},
            {transform * tf2::Vector3(-dimensions.x/2, dimensions.y/2, dimensions.z/2), classes},
            {transform * tf2::Vector3(dimensions.x/2, -dimensions.y/2, dimensions.z/2), classes},
            {transform * tf2::Vector3(dimensions.x/2, dimensions.y/2, dimensions.z/2), classes},
            // 其餘的邊界盒頂點也要添加
            };

            LidarDetection detection;
            detection.object.kinematics.initial_pose_with_covariance.pose.position = position;
            detection.object.kinematics.initial_pose_with_covariance.pose.orientation = orientation;
            detection.object.shape.dimensions = dimensions;
            detection.object.kinematics.initial_twist_with_covariance.twist.linear = twist_with_covariance;

            all_labeled_box_points.push_back(labeled_box_points);
            all_lidar_detection.push_back(detection);
        }
    } 
    else {
        RCLCPP_WARN(this->get_logger(), "No objects detected in object_msg");
    }
    
    all_radar_points_F.clear();
    all_radar_points_L.clear();
    all_radar_points_R.clear();

    if (!radar_msg1->returns.empty()) {
        geometry_msgs::msg::TransformStamped::ConstSharedPtr transform;

        velocity = odom_msg->twist.twist.linear.x;
        try {
            transform = transform_listener_->getTransform(
            odom_msg->header.frame_id, radar_msg1->header.frame_id, odom_msg->header.stamp,
            rclcpp::Duration::from_seconds(0.2));
        } catch (tf2::TransformException & ex) {
            RCLCPP_INFO(this->get_logger(), "Could not transform");
            return;
        }

        for (const auto & radar_return1 : radar_msg1->returns) {
            if (!isStaticPointcloud(radar_return1, odom_msg, transform)) {
                float Y = radar_return1.range * sin(radar_return1.azimuth);
                float X = radar_return1.range * cos(radar_return1.azimuth) + 2.3;
                float Z = radar_return1.range * sin(radar_return1.elevation) - 1.6;

                // 創建tf2::Vector3來存儲雷達點
                tf2::Vector3 radar_point(Y, X, Z);

                // 計算旋轉
                tf2::Quaternion q;
                q.setRPY(0, 0, M_PI * 1.5); // 1.5 * π 是 270 度

                // 進行座標轉換
                tf2::Transform transform(q);
                tf2::Vector3 transformed_point = transform * radar_point;

                std::vector<RadarPoint> radar_pointsF = {{transformed_point}};
                all_radar_points_F.push_back(radar_pointsF);
            }
        }
    }
    else {
        RCLCPP_WARN(this->get_logger(), "No radar detected in object_msg");
    }

    if (!radar_msg2->returns.empty()) {
        geometry_msgs::msg::TransformStamped::ConstSharedPtr transform;

        velocity = odom_msg->twist.twist.linear.x;
        nav_msgs::msg::Odometry::SharedPtr newOdometry = std::make_shared<nav_msgs::msg::Odometry>();
        *newOdometry = *odom_msg;
        newOdometry->twist.twist.linear.x = odom_msg->twist.twist.linear.x * -1;//轉180度
    
        try {
            transform = transform_listener_->getTransform(
            newOdometry->header.frame_id, radar_msg2->header.frame_id, newOdometry->header.stamp,
            rclcpp::Duration::from_seconds(0.2));
        } catch (tf2::TransformException & ex) {
            RCLCPP_INFO(this->get_logger(), "Could not transform");
            return;
        }

        for (const auto & radar_return2 : radar_msg2->returns) {
            if (!isStaticPointcloud(radar_return2, newOdometry, transform))
            {
                float Y = radar_return2.range * sin(radar_return2.azimuth) + 0.8;
                float X = radar_return2.range * cos(radar_return2.azimuth) + 2.3;
                float Z = radar_return2.range * sin(radar_return2.elevation) - 1.6;

                // 創建tf2::Vector3來存儲雷達點
                tf2::Vector3 radar_point(Y, X, Z);

                // 計算旋轉
                tf2::Quaternion q;
                q.setRPY(0, 0, M_PI * 0.5); // 1.5 * π 是 270 度

                // 進行座標轉換
                tf2::Transform transform(q);
                tf2::Vector3 transformed_point = transform * radar_point;

                std::vector<RadarPoint> radar_pointsL = {{transformed_point}};
                all_radar_points_L.push_back(radar_pointsL);
            }
        }
    }
    else {
        RCLCPP_WARN(this->get_logger(), "No radar detected in object_msg");
    }

    if (!radar_msg3->returns.empty()) {
        geometry_msgs::msg::TransformStamped::ConstSharedPtr transform;

        velocity = odom_msg->twist.twist.linear.x;
        nav_msgs::msg::Odometry::SharedPtr newOdometry = std::make_shared<nav_msgs::msg::Odometry>();
        *newOdometry = *odom_msg;
        newOdometry->twist.twist.linear.x = odom_msg->twist.twist.linear.x * -1;//轉180度
        
        try {
            transform = transform_listener_->getTransform(
            newOdometry->header.frame_id, radar_msg3->header.frame_id, newOdometry->header.stamp,
            rclcpp::Duration::from_seconds(0.2));
        } catch (tf2::TransformException & ex) {
            RCLCPP_INFO(this->get_logger(), "Could not transform");
            return;
        }

        for (const auto & radar_return3 : radar_msg3->returns) {
            if (!isStaticPointcloud(radar_return3, newOdometry, transform))
            {
                float Y = radar_return3.range * sin(radar_return3.azimuth) - 0.8;
                float X = radar_return3.range * cos(radar_return3.azimuth) + 2.3;
                float Z = radar_return3.range * sin(radar_return3.elevation + 3.2);

                // 創建tf2::Vector3來存儲雷達點
                tf2::Vector3 radar_point(Y, X, Z);

                // 計算旋轉
                tf2::Quaternion q;
                q.setRPY(0, 0, M_PI * 0.5); // 1.5 * π 是 270 度

                // 進行座標轉換
                tf2::Transform transform(q);
                tf2::Vector3 transformed_point = transform * radar_point;

                std::vector<RadarPoint> radar_pointsR = {{transformed_point}};
                all_radar_points_R.push_back(radar_pointsR);
            }
        }
    }
    else {
        RCLCPP_WARN(this->get_logger(), "No radar detected in object_msg");
    }
    
    // RCLCPP_INFO(this->get_logger(), "Timestamp for image 1: %d.%09d", in_roi_msg->header.stamp.sec, in_roi_msg->header.stamp.nanosec);
    std::array<tier4_perception_msgs::msg::DetectedObjectsWithFeature::ConstSharedPtr, 4> in_roi_msgs = {in_roi_msg, in_roi_msg1, in_roi_msg2, in_roi_msg3};

    
    FrontIPM = cv::getPerspectiveTransform(Front, dst60);
    LeftIPM = cv::getPerspectiveTransform(Left, dst140);
    RightIPM = cv::getPerspectiveTransform(Right, dst140);
    RearIPM = cv::getPerspectiveTransform(Rear, dst140);

    for (int i = 0; i < in_roi_msgs.size(); i++)
    {
        for (const auto &object : in_roi_msgs[i]->feature_objects) {
            points2f.clear();
            if(i==0){
                if(object.object.classification.front().label>0 && object.object.classification.front().label <=7){

                    points2f.push_back(cv::Point2f(object.feature.roi.x_offset, object.feature.roi.y_offset + object.feature.roi.height)); // 左下角
                    points2f.push_back(cv::Point2f(object.feature.roi.x_offset + object.feature.roi.width, object.feature.roi.y_offset + object.feature.roi.height)); // 右下角
                    // points2f.push_back(cv::Point2f(object.feature.roi.x_offset, object.feature.roi.y_offset + object.feature.roi.height)); // 左下角
                    // points2f.push_back(cv::Point2f(object.feature.roi.x_offset + object.feature.roi.width, object.feature.roi.y_offset + object.feature.roi.height)); // 右下角
                }
                
                if(points2f.size() > 0)
                {
                    transformedPoints.clear();
                    cv::perspectiveTransform(points2f, transformedPoints, FrontIPM);

                    for (size_t j = 0; j < transformedPoints.size(); ++j) {
                        // std::cout << transformedPoints[j].x << "  "  << transformedPoints[j].y <<std::endl;
                        // std::cout << "-----------------------" <<std::endl;
                        transformedPoints[j].y =  transformedPoints[j].y + 0.7;
                        classifiedPoints.push_back({transformedPoints[j], object.object.classification.front().label});
                    }
                }
            }
            if(i==1){
                if(object.object.classification.front().label>=0 && object.object.classification.front().label <=7){

                    points2f.push_back(cv::Point2f(object.feature.roi.x_offset, object.feature.roi.y_offset + object.feature.roi.height)); // 左下角
                    points2f.push_back(cv::Point2f(object.feature.roi.x_offset + object.feature.roi.width, object.feature.roi.y_offset + object.feature.roi.height)); // 右下角
                    // points2f.push_back(cv::Point2f(object.feature.roi.x_offset, object.feature.roi.y_offset + object.feature.roi.height)); // 左下角
                    // points2f.push_back(cv::Point2f(object.feature.roi.x_offset + object.feature.roi.width, object.feature.roi.y_offset + object.feature.roi.height)); // 右下角
                }
                
                if(points2f.size() > 0)
                {
                    transformedPoints.clear();
                    cv::perspectiveTransform(points2f, transformedPoints, LeftIPM);

                    for (size_t j = 0; j < transformedPoints.size(); ++j) {
                        // std::cout << -transformedPoints[j].y << "  "  << transformedPoints[j].x <<std::endl;
                        // std::cout << "-----------------------" <<std::endl;
                        float new_x = transformedPoints[j].y;
                        float new_y = -transformedPoints[j].x;
                        transformedPoints[j].x = new_x - 1;
                        transformedPoints[j].y = new_y + 0.3;
                        classifiedPoints.push_back({transformedPoints[j], object.object.classification.front().label});
                    }
                }
            }
            if(i==2){
                if(object.object.classification.front().label>0 && object.object.classification.front().label <=7){

                    points2f.push_back(cv::Point2f(object.feature.roi.x_offset, object.feature.roi.y_offset + object.feature.roi.height)); // 左下角
                    points2f.push_back(cv::Point2f(object.feature.roi.x_offset + object.feature.roi.width, object.feature.roi.y_offset + object.feature.roi.height)); // 右下角
                    // points2f.push_back(cv::Point2f(object.feature.roi.x_offset, object.feature.roi.y_offset + object.feature.roi.height)); // 左下角
                    // points2f.push_back(cv::Point2f(object.feature.roi.x_offset + object.feature.roi.width, object.feature.roi.y_offset + object.feature.roi.height)); // 右下角
                }
                
                if(points2f.size() > 0)
                {
                    transformedPoints.clear();
                    
                    cv::perspectiveTransform(points2f, transformedPoints, RightIPM);
                    for (size_t j = 0; j < transformedPoints.size(); ++j) {
                        // std::cout << -transformedPoints[j].y << "  "  << transformedPoints[j].x <<std::endl;
                        // std::cout << "-----------------------" <<std::endl;
                        float new_x = transformedPoints[j].y;
                        float new_y = -transformedPoints[j].x;
                        transformedPoints[j].x = new_x + 1;
                        transformedPoints[j].y = new_y + 0.3;
                        classifiedPoints.push_back({transformedPoints[j], object.object.classification.front().label});
                    }
                }
            }
            if(i==3){
                if(object.object.classification.front().label>0 && object.object.classification.front().label <=7){

                    points2f.push_back(cv::Point2f(object.feature.roi.x_offset, object.feature.roi.y_offset + object.feature.roi.height)); // 左下角
                    points2f.push_back(cv::Point2f(object.feature.roi.x_offset + object.feature.roi.width, object.feature.roi.y_offset + object.feature.roi.height)); // 右下角
                    // points2f.push_back(cv::Point2f(object.feature.roi.x_offset, object.feature.roi.y_offset + object.feature.roi.height)); // 左下角
                    // points2f.push_back(cv::Point2f(object.feature.roi.x_offset + object.feature.roi.width, object.feature.roi.y_offset + object.feature.roi.height)); // 右下角
                }
                
                if(points2f.size() > 0)
                {
                    transformedPoints.clear();
                    cv::perspectiveTransform(points2f, transformedPoints, RearIPM);

                    for (size_t j = 0; j < transformedPoints.size(); ++j) {
                        // std::cout << -transformedPoints[j].y << "  "  << transformedPoints[j].x <<std::endl;
                        // std::cout << "-----------------------" <<std::endl;
                        transformedPoints[j].y = (-1 * transformedPoints[j].y) - 2.3;
                        classifiedPoints.push_back({transformedPoints[j], object.object.classification.front().label});
                    }
                }
            }
        }
    }
    // RCLCPP_INFO(this->get_logger(), "Timestamp for image 1: %d.%09d", in_roi_msg->header.stamp.sec, in_roi_msg->header.stamp.nanosec);

    // 在Lidar_Detection_msg的if判斷之後，建立檔案名稱和開啟txt檔案
    std::string file_name = "/home/lewin/carla-autoware-universe/autoware/src/universe/autoware.universe/perception/sensor_fusion/result/" + std::to_string(in_roi_msg->header.stamp.sec) + "." + std::to_string(in_roi_msg->header.stamp.nanosec) + ".txt";
    std::ofstream outfile(file_name);
    // 假設有一個集合來存儲所有從LiDAR資料中發現的標籤
    lidar_labels.clear();
    PredictedObjects output;
    output.header = in_roi_msg->header;
    output.header.frame_id = "map";
    PredictedObject point_msg;
    PredictedObject min_point;
    PredictedObject lidar_objects;

    // std::cout<<"object_msg: " + std::to_string(object_msg->header.stamp.sec) + "." + std::to_string(object_msg->header.stamp.nanosec) << std::endl;
    // std::cout<<"in_roi_msg: " + std::to_string(in_roi_msg->header.stamp.sec) + "." + std::to_string(in_roi_msg->header.stamp.nanosec) << std::endl;

    if (outfile.is_open())
    {
        if(!all_radar_points_F.empty() || !all_radar_points_L.empty() || !all_radar_points_R.empty()){
            average_radar_points.clear();
            if(!all_radar_points_F.empty()){
                std::vector<cv::Point2f> temp_radar_points;
                bool is_first_point = true;
                for (const auto& radar_points : all_radar_points_F) {
                    for (const auto& radar_point : radar_points) {
                        outfile << "2 " << radar_point.point.x() << " " << radar_point.point.y() << " " << radar_point.point.z() << " " << -1 << std::endl;
                        processRadarPoint(cv::Point2f(radar_point.point.x(), radar_point.point.y()), temp_radar_points);
                        // 檢查是否為第一個點或是否有更小的x座標
                        if (is_first_point || radar_point.point.x() < min_point.kinematics.initial_pose_with_covariance.pose.position.x) {
                            min_point.kinematics.initial_pose_with_covariance.pose.position.x = radar_point.point.x();
                            min_point.kinematics.initial_pose_with_covariance.pose.position.y = radar_point.point.y();
                            min_point.kinematics.initial_pose_with_covariance.pose.position.z = radar_point.point.z();
                            is_first_point = false;
                        }
                    }
                }
                if (!temp_radar_points.empty()) {
                    cv::Point2f averagePoint = calculateAverage(temp_radar_points);
                    average_radar_points.push_back(averagePoint);
                }
                
                    // 只有當至少有一個點時才發布
                if (!is_first_point) {
                    output.objects.push_back(min_point);
                }
            }
            if(!all_radar_points_L.empty() && !all_radar_points_R.empty()){
                // 初始化一個變數來存儲x座標最小的點
                // PredictedObject min_point;
                // RCLCPP_INFO(this->get_logger(), "222");
                std::vector<cv::Point2f> temp_radar_points;
                bool is_first_point = true;
                for (const auto& radar_points : all_radar_points_L) {
                    for (const auto& radar_point : radar_points) {
                        outfile << "2 " << radar_point.point.x() << " " << radar_point.point.y() << " " << radar_point.point.z() << " " << -2 << std::endl;
                        processRadarPoint(cv::Point2f(radar_point.point.x(), radar_point.point.y()), temp_radar_points);                            // 檢查是否為第一個點或是否有更小的x座標
                        if (is_first_point || radar_point.point.x() < min_point.kinematics.initial_pose_with_covariance.pose.position.x) {
                            min_point.kinematics.initial_pose_with_covariance.pose.position.x = radar_point.point.x();
                            min_point.kinematics.initial_pose_with_covariance.pose.position.y = radar_point.point.y();
                            min_point.kinematics.initial_pose_with_covariance.pose.position.z = radar_point.point.z();
                            is_first_point = false;
                        }
                    }
                }
                for (const auto& radar_points : all_radar_points_R) {
                    for (const auto& radar_point : radar_points) {
                        outfile << "2 " << radar_point.point.x() << " " << radar_point.point.y() << " " << radar_point.point.z() << " " << -2 << std::endl;
                        processRadarPoint(cv::Point2f(radar_point.point.x(), radar_point.point.y()), temp_radar_points);                            // 檢查是否為第一個點或是否有更小的x座標
                        // 檢查是否為第一個點或是否有更小的x座標
                        if (is_first_point || radar_point.point.x() < min_point.kinematics.initial_pose_with_covariance.pose.position.x) {
                        min_point.kinematics.initial_pose_with_covariance.pose.position.x = radar_point.point.x();
                        min_point.kinematics.initial_pose_with_covariance.pose.position.y = radar_point.point.y();
                        min_point.kinematics.initial_pose_with_covariance.pose.position.z = radar_point.point.z();
                        is_first_point = false;
                        }
                    }
                }

                if (!temp_radar_points.empty()) {
                    cv::Point2f averagePoint = calculateAverage(temp_radar_points);
                    average_radar_points.push_back(averagePoint);
                }

                // 只有當至少有一個點時才發布
                if (!is_first_point) {
                    output.objects.push_back(min_point);
                }
            }
            else{
                if(!all_radar_points_L.empty()){
                    // 初始化一個變數來存儲x座標最小的點
                    // PredictedObject min_point;
                    // RCLCPP_INFO(this->get_logger(), "333");
                    std::vector<cv::Point2f> temp_radar_points;
                    bool is_first_point = true;
                    for (const auto& radar_points : all_radar_points_L) {
                        for (const auto& radar_point : radar_points) {
                            outfile << "2 " << radar_point.point.x() << " " << radar_point.point.y() << " " << radar_point.point.z() << " " << -2 << std::endl;
                            processRadarPoint(cv::Point2f(radar_point.point.x(), radar_point.point.y()), temp_radar_points);                            // 檢查是否為第一個點或是否有更小的x座標
                            // 檢查是否為第一個點或是否有更小的x座標
                            if (is_first_point || radar_point.point.x() < min_point.kinematics.initial_pose_with_covariance.pose.position.x) {
                            min_point.kinematics.initial_pose_with_covariance.pose.position.x = radar_point.point.x();
                            min_point.kinematics.initial_pose_with_covariance.pose.position.y = radar_point.point.y();
                            min_point.kinematics.initial_pose_with_covariance.pose.position.z = radar_point.point.z();
                            is_first_point = false;
                            }
                        }
                    }

                    if (!temp_radar_points.empty()) {
                        cv::Point2f averagePoint = calculateAverage(temp_radar_points);
                        average_radar_points.push_back(averagePoint);
                    }

                    // 只有當至少有一個點時才發布
                    if (!is_first_point) {
                        output.objects.push_back(min_point);
                    }
                }
                if(!all_radar_points_R.empty()){
                    // 初始化一個變數來存儲x座標最小的點
                    // PredictedObject min_point;
                    // RCLCPP_INFO(this->get_logger(), "333");
                    std::vector<cv::Point2f> temp_radar_points;
                    bool is_first_point = true;
                    for (const auto& radar_points : all_radar_points_R) {
                        for (const auto& radar_point : radar_points) {
                            outfile << "2 " << radar_point.point.x() << " " << radar_point.point.y() << " " << radar_point.point.z() << " " << -3 << std::endl;
                            processRadarPoint(cv::Point2f(radar_point.point.x(), radar_point.point.y()), temp_radar_points);                            // 檢查是否為第一個點或是否有更小的x座標
                            // 檢查是否為第一個點或是否有更小的x座標
                            if (is_first_point || radar_point.point.x() < min_point.kinematics.initial_pose_with_covariance.pose.position.x) {
                            min_point.kinematics.initial_pose_with_covariance.pose.position.x = radar_point.point.x();
                            min_point.kinematics.initial_pose_with_covariance.pose.position.y = radar_point.point.y();
                            min_point.kinematics.initial_pose_with_covariance.pose.position.z = radar_point.point.z();
                            is_first_point = false;
                            }
                        }
                    }

                    if (!temp_radar_points.empty()) {
                        cv::Point2f averagePoint = calculateAverage(temp_radar_points);
                        average_radar_points.push_back(averagePoint);
                    }

                    // 只有當至少有一個點時才發布
                    if (!is_first_point) {
                        output.objects.push_back(min_point);
                    }
                }
            }

            if(!classifiedPoints.empty()) {
                // 存取bounding box的資料
                // PredictedObject point_msg;
                int label = 0, check = 0;
                float new_x = 0, new_y = 0;
                float pre_x = 0, pre_y = 0;

                for (const auto& classifiedPoint : classifiedPoints) {
                    if(classifiedPoint.classId>0 && classifiedPoint.classId<=4){
                        label = 2;
                    }
                    else if(classifiedPoint.classId>=5 && classifiedPoint.classId<=6){
                        label = 1;
                    }
                    else if(classifiedPoint.classId==7){
                        label = 0;
                    }
                    
                    new_x += classifiedPoint.point.y;
                    new_y += -classifiedPoint.point.x;

                    if(check % 2 ==0){
                        pre_x = classifiedPoint.point.y;
                        pre_y = -classifiedPoint.point.x;
                    }

                    ++check;

                    if(check % 2 == 0){
                        cv::Point2f centerPoint(new_x / 2, new_y / 2);
                        std::cout << centerPoint << std::endl;
                        if(isPointInRadarRange(centerPoint, average_radar_points, 5.0)){
                            outfile << "0 " << pre_x << " " << pre_y << " " << "0" << " " << label << std::endl;
                            outfile << "0 " << classifiedPoint.point.y << " " << -classifiedPoint.point.x << " " << "0" << " " << label << std::endl;
                            point_msg.kinematics.initial_pose_with_covariance.pose.position.x = new_x/2;
                            point_msg.kinematics.initial_pose_with_covariance.pose.position.y = new_y/2;
                            // point_msg.kinematics.initial_pose_with_covariance.pose.orientation =
                            //   tier4_autoware_utils::createQuaternionFromYaw(0.0);
                            // point_msg.kinematics.initial_twist_with_covariance.twist.linear.x = 0.0;
                            point_msg.shape.dimensions.x = 1.0;
                            point_msg.shape.dimensions.y = 1.0;
                            output.objects.push_back(point_msg);
                            new_x = 0;
                            new_y = 0;
                            pre_x = 0;
                            pre_y = 0;
                        }
                    }
                }
            }
        }
        else if(!classifiedPoints.empty()){
            // 存取bounding box的資料
            // PredictedObject point_msg;
            int label = 0;
            float new_x = 0;
            float new_y = 0;
            int check = 0;
            for (const auto& classifiedPoint : classifiedPoints) {
                if(classifiedPoint.classId>0 && classifiedPoint.classId<=4){
                    label = 2;
                }
                else if(classifiedPoint.classId>=5 && classifiedPoint.classId<=6){
                    label = 1;
                }
                else if(classifiedPoint.classId==7){
                    label = 0;
                }
                
                outfile << "0 " << classifiedPoint.point.y << " " << -classifiedPoint.point.x << " " << "0" << " " << label << std::endl;

                new_x += classifiedPoint.point.y;
                new_y += -classifiedPoint.point.x;
                check++;

                if(check == 2 ){
                    point_msg.kinematics.initial_pose_with_covariance.pose.position.x = new_x/2;
                    point_msg.kinematics.initial_pose_with_covariance.pose.position.y = new_y/2;
                    // point_msg.kinematics.initial_pose_with_covariance.pose.orientation =
                    //   tier4_autoware_utils::createQuaternionFromYaw(0.0);
                    // point_msg.kinematics.initial_twist_with_covariance.twist.linear.x = 0.0;
                    point_msg.shape.dimensions.x = 1.0;
                    point_msg.shape.dimensions.y = 1.0;
                    output.objects.push_back(point_msg);
                    new_x = 0;
                    new_y = 0;
                    check = 0;
                }
            }
        }
        else{
            if (!all_labeled_box_points.empty()) {
                // LiDAR有輸出，所以我們只存取LiDAR的點資料
                for (const auto& labeled_box_points : all_labeled_box_points) {
                for (const auto& labeled_point : labeled_box_points) {
                    outfile << "1 " << labeled_point.point.x() << " " << labeled_point.point.y() << " " << labeled_point.point.z() << " " << labeled_point.label << std::endl;
                    lidar_labels.insert(labeled_point.label); // 將標籤加入到集合中
                }
                }
                for(const auto& lidar_detection : all_lidar_detection){
                lidar_objects.kinematics.initial_pose_with_covariance.pose.position.x = lidar_detection.object.kinematics.initial_pose_with_covariance.pose.position.x;
                lidar_objects.kinematics.initial_pose_with_covariance.pose.position.y = lidar_detection.object.kinematics.initial_pose_with_covariance.pose.position.y;
                lidar_objects.kinematics.initial_pose_with_covariance.pose.orientation = lidar_detection.object.kinematics.initial_pose_with_covariance.pose.orientation;
                lidar_objects.kinematics.initial_twist_with_covariance.twist.linear.x = lidar_detection.object.kinematics.initial_twist_with_covariance.twist.linear.x;
                lidar_objects.shape.dimensions.x = lidar_detection.object.shape.dimensions.x;
                lidar_objects.shape.dimensions.y = lidar_detection.object.shape.dimensions.y;
                output.objects.push_back(lidar_objects);
                }

                // 如果LiDAR有輸出，再檢查bounding box資料，並只存取那些在LiDAR資料中沒有的標籤
                if (!lidar_labels.empty()) {
                // PredictedObject point_msg;
                    for (const auto& classifiedPoint : classifiedPoints) {
                        // 如果LiDAR資料集合中沒有這個標籤，則將其寫入文件
                        int label = 0;
                        float new_x = 0;
                        float new_y = 0;
                        int check = 0;
                        for (const auto& classifiedPoint : classifiedPoints) {
                            if(classifiedPoint.classId>0 && classifiedPoint.classId<=4){
                                label = 2;
                            }
                            else if(classifiedPoint.classId>=5 && classifiedPoint.classId<=6){
                                label = 1;
                            }
                            else if(classifiedPoint.classId==7){
                                label = 0;
                            }
                            if (lidar_labels.find(classifiedPoint.classId) == lidar_labels.end()) {
                                outfile << "0 " << classifiedPoint.point.y << " " << -classifiedPoint.point.x << " " << "0" << " " << label << std::endl;

                                new_x += classifiedPoint.point.y;
                                new_y += -classifiedPoint.point.x;
                                check++;

                                if(check == 2 ){
                                    point_msg.kinematics.initial_pose_with_covariance.pose.position.x = new_x/2;
                                    point_msg.kinematics.initial_pose_with_covariance.pose.position.y = new_y/2;
                                    // point_msg.kinematics.initial_pose_with_covariance.pose.orientation =
                                    //   tier4_autoware_utils::createQuaternionFromYaw(0.0);
                                    // point_msg.kinematics.initial_twist_with_covariance.twist.linear.x = 0.0;
                                    point_msg.shape.dimensions.x = 1.0;
                                    point_msg.shape.dimensions.y = 1.0;
                                    output.objects.push_back(point_msg);
                                    new_x = 0;
                                    new_y = 0;
                                    check = 0;
                                }
                            }
                        }
                    }
                }
            } 
        }
        // Publish Results
        pub_objects_->publish(output);
        outfile.close();
    } 
    else 
    {
        RCLCPP_ERROR(this->get_logger(), "無法打開檔案: %s", file_name.c_str());
    }
    classifiedPoints.clear();

}

SensorFusionNodelet::~SensorFusionNodelet()
{
}

bool object_recognition::SensorFusionNodelet::isStaticPointcloud(
  const RadarReturn & radar_return, const Odometry::ConstSharedPtr & odom_msg,
  geometry_msgs::msg::TransformStamped::ConstSharedPtr transform)
{
  geometry_msgs::msg::Vector3 compensated_velocity =
    compensateEgoVehicleTwist(radar_return, odom_msg->twist, transform);

  return (-node_param_.doppler_velocity_sd < compensated_velocity.x) &&
         (compensated_velocity.x < node_param_.doppler_velocity_sd);
}

}  // namespace object_recognition

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(object_recognition::SensorFusionNodelet)
