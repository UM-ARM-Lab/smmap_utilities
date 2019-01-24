#ifndef VISUALIZATION_TOOLS_H
#define VISUALIZATION_TOOLS_H

#include <thread>
#include <mutex>

#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>
#include <deformable_manipulation_experiment_params/ros_params.hpp>
#include <arc_utilities/eigen_helpers.hpp>

namespace smmap_utilities
{
    class Visualizer
    {
        // TODO: Move these elsewhere, potentially to a location that is then used by smmap/trajectory.hpp
        typedef Eigen::Matrix3Xd ObjectPointSet;

        public:
            static void InitializeStandardColors();
            static std_msgs::ColorRGBA Red(const float alpha = 1.0f);
            static std_msgs::ColorRGBA Green(const float alpha = 1.0f);
            static std_msgs::ColorRGBA Blue(const float alpha = 1.0f);
            static std_msgs::ColorRGBA Black(const float alpha = 1.0f);
            static std_msgs::ColorRGBA Magenta(const float alpha = 1.0f);
            static std_msgs::ColorRGBA Yellow(const float alpha = 1.0f);
            static std_msgs::ColorRGBA Cyan(const float alpha = 1.0f);
            static std_msgs::ColorRGBA White(const float alpha = 1.0f);
            static std_msgs::ColorRGBA Silver(const float alpha = 1.0f);
            static std_msgs::ColorRGBA Coral(const float alpha = 1.0f);
            static std_msgs::ColorRGBA Olive(const float alpha = 1.0f);
            static std_msgs::ColorRGBA Orange(const float alpha = 1.0f);

        public:
            typedef std::shared_ptr<Visualizer> Ptr;
            typedef std::shared_ptr<const Visualizer> ConstPtr;

            Visualizer(
                    std::shared_ptr<ros::NodeHandle> nh,
                    std::shared_ptr<ros::NodeHandle> ph,
                    const bool publish_async = false);

            void publish(const visualization_msgs::Marker& marker) const;

            void publish(const visualization_msgs::MarkerArray& marker_array) const;

            void forcePublishNow(const double last_delay = 0.01) const;

            void clearVisualizationsBullet();

            void purgeMarkerList() const;

            // Converts all markers in the async publishing list to "delete" actions
            void deleteAll() const;

            // Publishes markers in namespace marker_name, with ids in the half close range [start_id, end_id)
            // with the delete flag set
            void deleteObjects(
                    const std::string& marker_name,
                    const int32_t start_id = 0,
                    const int32_t end_id = 1024) const;

            void visualizePoints(
                    const std::string& marker_name,
                    const EigenHelpers::VectorVector3d& points,
                    const std_msgs::ColorRGBA& color,
                    const int32_t id = 1,
                    const double scale = 0.005) const;

            void visualizePoints(
                    const std::string& marker_name,
                    const EigenHelpers::VectorVector3d& points,
                    const std::vector<std_msgs::ColorRGBA>& colors,
                    const int32_t id = 1,
                    const double scale = 0.005) const;

            void visualizeCubes(
                    const std::string& marker_name,
                    const EigenHelpers::VectorVector3d& points,
                    const Eigen::Vector3d& scale,
                    const std::vector<std_msgs::ColorRGBA>& colors,
                    const int32_t id = 1) const;

            void visualizeCubes(
                    const std::string& marker_name,
                    const EigenHelpers::VectorVector3d& points,
                    const Eigen::Vector3d& scale,
                    const std_msgs::ColorRGBA& color,
                    const int32_t id = 1) const;

            void visualizeSpheres(const std::string& marker_name,
                    const EigenHelpers::VectorVector3d& points,
                    const std_msgs::ColorRGBA& color,
                    const int32_t starting_id,
                    const double& radius) const;

            void visualizeSpheres(const std::string& marker_name,
                    const EigenHelpers::VectorVector3d& points,
                    const std_msgs::ColorRGBA& color,
                    const int32_t starting_id,
                    const std::vector<double>& radiuses) const;

            void visualizeSpheres(const std::string& marker_name,
                    const EigenHelpers::VectorVector3d& points,
                    const std::vector<std_msgs::ColorRGBA>& colors,
                    const int32_t starting_id,
                    const std::vector<double>& radiuses) const;

            void visualizeRope(
                    const std::string& marker_name,
                    const ObjectPointSet& rope,
                    const std_msgs::ColorRGBA& color,
                    const int32_t id = 1) const;

            void visualizeRope(
                    const std::string& marker_name,
                    const ObjectPointSet& rope,
                    const std::vector<std_msgs::ColorRGBA>& colors,
                    const int32_t id = 1) const;

            void visualizeCloth(
                    const std::string& marker_name,
                    const ObjectPointSet& cloth,
                    const std_msgs::ColorRGBA& color,
                    const int32_t id = 1) const;

            void visualizeCloth(
                    const std::string& marker_name,
                    const ObjectPointSet& cloth,
                    const std::vector<std_msgs::ColorRGBA>& colors,
                    const int32_t id = 1) const;

            void visualizeGripper(
                    const std::string& marker_name,
                    const Eigen::Isometry3d& eigen_pose,
                    const std_msgs::ColorRGBA& color,
                    const int32_t id = 1) const;

            void visualizeGrippers(
                    const std::string& marker_name,
                    const EigenHelpers::VectorIsometry3d eigen_poses,
                    const std_msgs::ColorRGBA& color,
                    const int32_t id = 1) const;

            void visualizeObjectDelta(
                    const std::string& marker_name,
                    const ObjectPointSet& current,
                    const ObjectPointSet& desired,
                    const std_msgs::ColorRGBA& color,
                    const int32_t id = 2) const;

            void visualizeTranslation(
                    const std::string& marker_name,
                    const geometry_msgs::Point& start,
                    const geometry_msgs::Point& end,
                    const std_msgs::ColorRGBA& color,
                    const int32_t id = 3) const;

            void visualizeTranslation(
                    const std::string& marker_name,
                    const Eigen::Vector3d& start,
                    const Eigen::Vector3d& end,
                    const std_msgs::ColorRGBA& color,
                    const int32_t id = 3) const;

            void visualizeTranslation(
                    const std::string& marker_name,
                    const Eigen::Isometry3d &start,
                    const Eigen::Isometry3d &end,
                    const std_msgs::ColorRGBA& color,
                    const int32_t id = 3) const;

            void visualizeArrows(
                    const std::string& marker_name,
                    const EigenHelpers::VectorVector3d& start,
                    const EigenHelpers::VectorVector3d& end,
                    const std_msgs::ColorRGBA& color,
                    const double scale_x,
                    const double scale_y,
                    const double scale_z = 0.0,
                    const int32_t start_id = 1) const;

            void visualizeLines(
                    const std::string& marker_name,
                    const EigenHelpers::VectorVector3d& start,
                    const EigenHelpers::VectorVector3d& end,
                    const std_msgs::ColorRGBA& color,
                    const int32_t id = 1,
                    const double scale = 0.001) const;

            void visualizeLines(
                    const std::string& marker_name,
                    const EigenHelpers::VectorVector3d& start,
                    const EigenHelpers::VectorVector3d& end,
                    const std::vector<std_msgs::ColorRGBA>& colors,
                    const int32_t id = 1,
                    const double scale = 0.001) const;

            void visualizeLineStrip(
                    const std::string& marker_name,
                    const EigenHelpers::VectorVector3d& point_sequence,
                    const std_msgs::ColorRGBA& color,
                    const int32_t id = 1,
                    const double scale = 0.001) const;

            void visualizeXYZTrajectory(
                    const std::string& marker_name,
                    const EigenHelpers::VectorVector3d& point_sequence,
                    const std_msgs::ColorRGBA& color,
                    const int32_t id = 1) const;

        private:

            void updateMarkerList(const visualization_msgs::Marker& marker) const;
            void publishAsyncMain();

            const std::shared_ptr<ros::NodeHandle> nh_;
            const std::shared_ptr<ros::NodeHandle> ph_;

            const bool publish_async_;
            std::thread publish_thread_;
            mutable std::mutex markers_mtx_;
            mutable visualization_msgs::MarkerArray async_markers_;

            const bool disable_all_visualizations_;
            ros::ServiceClient clear_markers_srv_;
            ros::Publisher visualization_marker_pub_;
            ros::Publisher visualization_maker_array_pub_;

            // Data needed to properly create visualizations and markers
            const std::string world_frame_name_;
            const double gripper_apperture_;

            static bool standard_colors_initialized_;
            static std_msgs::ColorRGBA red_;
            static std_msgs::ColorRGBA green_;
            static std_msgs::ColorRGBA blue_;
            static std_msgs::ColorRGBA black_;
            static std_msgs::ColorRGBA magenta_;
            static std_msgs::ColorRGBA yellow_;
            static std_msgs::ColorRGBA cyan_;
            static std_msgs::ColorRGBA white_;
            static std_msgs::ColorRGBA silver_;
            static std_msgs::ColorRGBA coral_;
            static std_msgs::ColorRGBA olive_;
            static std_msgs::ColorRGBA orange_;
    };
}

#endif // VISUALIZATION_TOOLS_H
