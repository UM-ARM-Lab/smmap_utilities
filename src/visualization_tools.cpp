#include "smmap_utilities/visualization_tools.h"

#include <thread>
#include <std_srvs/Empty.h>
#include <arc_utilities/eigen_helpers_conversions.hpp>

using namespace smmap_utilities;

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////

bool Visualizer::standard_colors_initialized_ = false;
std_msgs::ColorRGBA Visualizer::red_;
std_msgs::ColorRGBA Visualizer::green_;
std_msgs::ColorRGBA Visualizer::blue_;
std_msgs::ColorRGBA Visualizer::black_;
std_msgs::ColorRGBA Visualizer::magenta_;
std_msgs::ColorRGBA Visualizer::yellow_;
std_msgs::ColorRGBA Visualizer::cyan_;
std_msgs::ColorRGBA Visualizer::white_;
std_msgs::ColorRGBA Visualizer::silver_;
std_msgs::ColorRGBA Visualizer::coral_;
std_msgs::ColorRGBA Visualizer::olive_;
std_msgs::ColorRGBA Visualizer::orange_;

void Visualizer::InitializeStandardColors()
{
    red_.r = 1.0;
    red_.g = 0.0;
    red_.b = 0.0;
    red_.a = 1.0;

    green_.r = 0.0;
    green_.g = 1.0;
    green_.b = 0.0;
    green_.a = 1.0;

    blue_.r = 0.0;
    blue_.g = 0.0;
    blue_.b = 1.0;
    blue_.a = 1.0;

    black_.r = 0.0;
    black_.g = 0.0;
    black_.b = 0.0;
    black_.a = 1.0;

    magenta_.r = 1.0f;
    magenta_.g = 0.0f;
    magenta_.b = 1.0f;
    magenta_.a = 1.0f;

    yellow_.r = 1.0f;
    yellow_.g = 1.0f;
    yellow_.b = 0.0f;
    yellow_.a = 1.0f;

    cyan_.r = 0.0f;
    cyan_.g = 1.0f;
    cyan_.b = 1.0f;
    cyan_.a = 1.0f;

    white_.r = 1.0f;
    white_.g = 1.0f;
    white_.b = 1.0f;
    white_.a = 1.0f;

    silver_.r = 0.75f;
    silver_.g = 0.75f;
    silver_.b = 0.75f;
    silver_.a = 1.0f;

    coral_.r = 0.8f;
    coral_.g = 0.36f;
    coral_.b = 0.27f;
    coral_.a = 1.0f;

    olive_.r = 0.31f;
    olive_.g = 0.31f;
    olive_.b = 0.18f;
    olive_.a = 1.0f;

    orange_.r = 0.8f;
    orange_.g = 0.2f;
    orange_.b = 0.2f;
    orange_.a = 1.0f;

    standard_colors_initialized_ = true;
}

std_msgs::ColorRGBA Visualizer::Red(const float alpha)
{
    assert(standard_colors_initialized_);
    auto color = red_;
    color.a = alpha;
    return color;
}

std_msgs::ColorRGBA Visualizer::Green(const float alpha)
{
    assert(standard_colors_initialized_);
    auto color = green_;
    color.a = alpha;
    return color;
}

std_msgs::ColorRGBA Visualizer::Blue(const float alpha)
{
    assert(standard_colors_initialized_);
    auto color = blue_;
    color.a = alpha;
    return color;
}

std_msgs::ColorRGBA Visualizer::Black(const float alpha)
{
    assert(standard_colors_initialized_);
    auto color = black_;
    color.a = alpha;
    return color;
}

std_msgs::ColorRGBA Visualizer::Magenta(const float alpha)
{
    assert(standard_colors_initialized_);
    auto color = magenta_;
    color.a = alpha;
    return color;
}

std_msgs::ColorRGBA Visualizer::Yellow(const float alpha)
{
    assert(standard_colors_initialized_);
    auto color = yellow_;
    color.a = alpha;
    return color;
}

std_msgs::ColorRGBA Visualizer::Cyan(const float alpha)
{
    assert(standard_colors_initialized_);
    auto color = cyan_;
    color.a = alpha;
    return color;
}

std_msgs::ColorRGBA Visualizer::White(const float alpha)
{
    assert(standard_colors_initialized_);
    auto color = white_;
    color.a = alpha;
    return color;
}

std_msgs::ColorRGBA Visualizer::Silver(const float alpha)
{
    assert(standard_colors_initialized_);
    auto color = silver_;
    color.a = alpha;
    return color;
}

std_msgs::ColorRGBA Visualizer::Coral(const float alpha)
{
    assert(standard_colors_initialized_);
    auto color = coral_;
    color.a = alpha;
    return color;
}

std_msgs::ColorRGBA Visualizer::Olive(const float alpha)
{
    assert(standard_colors_initialized_);
    auto color = olive_;
    color.a = alpha;
    return color;
}

std_msgs::ColorRGBA Visualizer::Orange(const float alpha)
{
    assert(standard_colors_initialized_);
    auto color = orange_;
    color.a = alpha;
    return color;
}

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////

Visualizer::Visualizer(
        ros::NodeHandle& nh,
        ros::NodeHandle& ph,
        const bool publish_async)
    : nh_(nh)
    , ph_(ph)
    , publish_async_(publish_async)
    , disable_all_visualizations_(smmap::GetDisableAllVisualizations(ph_))
    , clear_markers_srv_(nh_.serviceClient<std_srvs::Empty>(smmap::GetClearVisualizationsTopic(nh_), true))
    , world_frame_name_(smmap::GetWorldFrameName())
    , gripper_apperture_(smmap::GetGripperApperture(nh_))
{
    InitializeStandardColors();
    if (!disable_all_visualizations_)
    {
        clear_markers_srv_.waitForExistence();
        visualization_marker_pub_ = nh.advertise<visualization_msgs::Marker>(smmap::GetVisualizationMarkerTopic(nh_), 256);
        visualization_maker_array_pub_ = nh.advertise<visualization_msgs::MarkerArray>(smmap::GetVisualizationMarkerArrayTopic(nh_), 1);

        if (publish_async_)
        {
            publish_thread_ = std::thread(&Visualizer::publishAsyncMain, this);
        }
    }
}

void Visualizer::publish(const visualization_msgs::Marker& marker) const
{
    if (!disable_all_visualizations_)
    {
        if (publish_async_)
        {
            std::lock_guard<std::mutex> lock(markers_mtx_);

            bool marker_found = false;
            for (size_t idx = 0; idx < async_markers_.markers.size(); ++idx)
            {
                visualization_msgs::Marker& old_marker = async_markers_.markers[idx];
                if (old_marker.id == marker.id && old_marker.ns == marker.ns)
                {
                    old_marker = marker;
                    marker_found = true;
                    break;
                }
            }

            if (!marker_found)
            {
                async_markers_.markers.push_back(marker);
            }
        }
        else
        {
            visualization_marker_pub_.publish(marker);
        }
    }
}

void Visualizer::forcePublishNow(const double last_delay) const
{
    if (!disable_all_visualizations_)
    {
        if (publish_async_)
        {
            std::lock_guard<std::mutex> lock(markers_mtx_);
            visualization_maker_array_pub_.publish(async_markers_);
            visualization_maker_array_pub_.publish(async_markers_);
            visualization_maker_array_pub_.publish(async_markers_);
            visualization_maker_array_pub_.publish(async_markers_);
            visualization_maker_array_pub_.publish(async_markers_);
            ros::spinOnce();
            arc_helpers::Sleep(0.01);

            visualization_maker_array_pub_.publish(async_markers_);
            visualization_maker_array_pub_.publish(async_markers_);
            visualization_maker_array_pub_.publish(async_markers_);
            visualization_maker_array_pub_.publish(async_markers_);
            visualization_maker_array_pub_.publish(async_markers_);
            ros::spinOnce();
            arc_helpers::Sleep(0.01);

            arc_helpers::Sleep(last_delay);

//            ros::Rate rate(0.001);
//            const auto start = std::chrono::steady_clock::now();
//            while (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() < last_delay)
//            {
//                rate.sleep();
//            }
        }
        else
        {
            ROS_WARN_THROTTLE_NAMED(1.0, "visualizer", "forcePublishNow() does nothing if async publishing is not enabled.");
        }
    }
}

void Visualizer::clearVisualizationsBullet()
{
    if (!disable_all_visualizations_)
    {
        std_srvs::Empty srv_data;
        while (!clear_markers_srv_.call(srv_data))
        {
            ROS_WARN_THROTTLE_NAMED(1.0, "visualizer", "Clear visualization data failed, reconnecting");
            clear_markers_srv_ = nh_.serviceClient<std_srvs::Empty>(smmap::GetClearVisualizationsTopic(nh_), true);
            clear_markers_srv_.waitForExistence();
        }
    }
}

void Visualizer::deleteAll() const
{
    if (!disable_all_visualizations_)
    {
        if (publish_async_)
        {
            std::lock_guard<std::mutex> lock(markers_mtx_);
            for (size_t idx = 0; idx < async_markers_.markers.size(); ++idx)
            {
                visualization_msgs::Marker& marker = async_markers_.markers[idx];
                marker.action = visualization_msgs::Marker::DELETE;
                marker.header.stamp = ros::Time::now();
                marker.lifetime = ros::Duration(0.1);
                marker.points.clear();
                marker.colors.clear();
            }
        }
        else
        {
            ROS_WARN_THROTTLE_NAMED(1.0, "visualizer", "Visualizer::deleteAll() called when publishing synchronously; no marker data is stored in this mode, so no markers will be deleted. Use Visualizer::deleteObjects(...) to specify which objects to delete.");
        }
    }
}

void Visualizer::purgeMarkerList() const
{
    if (!disable_all_visualizations_)
    {
        if (publish_async_)
        {
            std::lock_guard<std::mutex> lock(markers_mtx_);
            async_markers_.markers.clear();
        }
        else
        {
            ROS_WARN_THROTTLE_NAMED(1.0, "visualizer", "purgeMarkerList() does nothing if async publishing is not enabled.");
        }
    }
}

void Visualizer::deleteObjects(
        const std::string& marker_name,
        const int32_t start_id,
        const int32_t end_id) const
{
    if (!disable_all_visualizations_)
    {
        visualization_msgs::Marker marker;
        marker.header.frame_id = world_frame_name_;
        marker.action = visualization_msgs::Marker::DELETE;
        marker.ns = marker_name;
        marker.lifetime = ros::Duration(1.0);

        if (publish_async_)
        {
            std::lock_guard<std::mutex> lock(markers_mtx_);

            // Flag any existing markers for deletion
            std::vector<size_t> markers_to_delete;
            for (size_t idx = 0; idx < async_markers_.markers.size(); ++idx)
            {
                const visualization_msgs::Marker& marker = async_markers_.markers[idx];
                if (marker.ns == marker_name &&
                    start_id <= marker.id &&
                    marker.id < end_id)
                {
                    markers_to_delete.push_back(idx);
                }
            }

            // Delete the flaged markers
            visualization_msgs::MarkerArray new_markers;
            new_markers.markers.reserve(async_markers_.markers.size() + end_id - start_id);
            for (size_t idx = 0; idx < async_markers_.markers.size(); ++idx)
            {
                const auto itr = std::find(markers_to_delete.begin(), markers_to_delete.end(), idx);
                if (itr != markers_to_delete.end())
                {
                    new_markers.markers.push_back(async_markers_.markers[idx]);
                }
            }
            async_markers_ = new_markers;

            // Add new "DELETE" markers
            for (int32_t id = start_id; id < end_id; ++id)
            {
                marker.id = id;
                marker.header.stamp = ros::Time::now();
                async_markers_.markers.push_back(marker);
            }
        }
        else
        {
            for (int32_t id = start_id; id < end_id; ++id)
            {
                marker.id = id;
                marker.header.stamp = ros::Time::now();
                publish(marker);

                if (id % 100 == 0)
                {
                    ros::spinOnce();
                    std::this_thread::sleep_for(std::chrono::duration<double>(0.001));
                }
            }

            ros::spinOnce();
            std::this_thread::sleep_for(std::chrono::duration<double>(0.001));
        }
    }
}

void Visualizer::visualizePoints(
        const std::string& marker_name,
        const EigenHelpers::VectorVector3d& points,
        const std_msgs::ColorRGBA& color,
        const int32_t id,
        const double scale) const
{
    if (!disable_all_visualizations_)
    {
        const std::vector<std_msgs::ColorRGBA> colors(points.size(), color);
        visualizePoints(marker_name, points, colors, id, scale);
    }
}

void Visualizer::visualizePoints(
        const std::string& marker_name,
        const EigenHelpers::VectorVector3d& points,
        const std::vector<std_msgs::ColorRGBA>& colors,
        const int32_t id,
        const double scale) const
{
    if (!disable_all_visualizations_)
    {
        visualization_msgs::Marker marker;

        marker.header.frame_id = world_frame_name_;

        marker.type = visualization_msgs::Marker::POINTS;
        marker.ns = marker_name;
        marker.id = id;
        marker.scale.x = scale;
        marker.scale.y = scale;
        marker.points = EigenHelpersConversions::VectorEigenVector3dToVectorGeometryPoint(points);
        marker.colors = colors;

        // Assumes that all non specified values are 0.0
        marker.pose.orientation.w = 1.0;

        marker.header.stamp = ros::Time::now();
        publish(marker);
    }
}

void Visualizer::visualizeCubes(
        const std::string& marker_name,
        const EigenHelpers::VectorVector3d& points,
        const Eigen::Vector3d& scale,
        const std::vector<std_msgs::ColorRGBA>& colors,
        const int32_t id) const
{
    if (!disable_all_visualizations_)
    {
        visualization_msgs::Marker marker;

        marker.header.frame_id = world_frame_name_;

        marker.type = visualization_msgs::Marker::CUBE_LIST;
        marker.ns = marker_name;
        marker.id = id;
        marker.scale = EigenHelpersConversions::EigenVector3dToGeometryVector3(scale);
        marker.points = EigenHelpersConversions::VectorEigenVector3dToVectorGeometryPoint(points);
        marker.colors = colors;

        // Assumes that all non specified values are 0.0
        marker.pose.orientation.w = 1.0;

        marker.header.stamp = ros::Time::now();
        publish(marker);
    }
}

void Visualizer::visualizeCubes(
        const std::string& marker_name,
        const EigenHelpers::VectorVector3d& points,
        const Eigen::Vector3d& scale,
        const std_msgs::ColorRGBA& color,
        const int32_t id) const
{
    if (!disable_all_visualizations_)
    {
        const std::vector<std_msgs::ColorRGBA> colors(points.size(), color);
        visualizeCubes(marker_name, points, scale, colors, id);
    }
}

void Visualizer::visualizeSpheres(const std::string& marker_name,
        const EigenHelpers::VectorVector3d& points,
        const std_msgs::ColorRGBA& color,
        const int32_t starting_id,
        const double& radius) const
{
    if (!disable_all_visualizations_)
    {
        const std::vector<std_msgs::ColorRGBA> colors(points.size(), color);
        const std::vector<double> radiuses(points.size(), radius);
        visualizeSpheres(marker_name, points, colors, starting_id, radiuses);
    }
}

void Visualizer::visualizeSpheres(
        const std::string& marker_name,
        const EigenHelpers::VectorVector3d& points,
        const std_msgs::ColorRGBA& color,
        const int32_t starting_id,
        const std::vector<double>& radiuses) const
{
    if (!disable_all_visualizations_)
    {
        const std::vector<std_msgs::ColorRGBA> colors(points.size(), color);
        visualizeSpheres(marker_name, points, colors, starting_id, radiuses);
    }
}

void Visualizer::visualizeSpheres(
        const std::string& marker_name,
        const EigenHelpers::VectorVector3d& points,
        const std::vector<std_msgs::ColorRGBA>& colors,
        const int32_t starting_id,
        const std::vector<double>& radiuses) const
{
    if (!disable_all_visualizations_)
    {
        if (points.size() != radiuses.size())
        {
            ROS_ERROR_NAMED("visualizer", "Invalid sphere list, need number of points and radiuses to match");
        }

        visualization_msgs::Marker marker;
        marker.header.frame_id = world_frame_name_;
        marker.header.stamp = ros::Time::now();

        for (size_t idx = 0; idx < points.size(); ++idx)
        {
            marker.type = visualization_msgs::Marker::SPHERE;
            marker.ns = marker_name;
            marker.id = starting_id + (int32_t)idx;
            marker.scale.x = radiuses[idx] * 2.0;
            marker.scale.y = radiuses[idx] * 2.0;
            marker.scale.z = radiuses[idx] * 2.0;
            marker.pose.position = EigenHelpersConversions::EigenVector3dToGeometryPoint(points[idx]);
            marker.pose.orientation.x = 0.0;
            marker.pose.orientation.y = 0.0;
            marker.pose.orientation.z = 0.0;
            marker.pose.orientation.w = 1.0;
            marker.color = colors[idx];

            publish(marker);
        }
    }
}

void Visualizer::visualizeRope(
        const std::string& marker_name,
        const ObjectPointSet& rope,
        const std_msgs::ColorRGBA& color,
        const int32_t id) const
{
    if (!disable_all_visualizations_)
    {
        const std::vector<std_msgs::ColorRGBA> colors((size_t)rope.cols(), color);
        visualizeRope(marker_name, rope, colors, id);
    }
}

void Visualizer::visualizeRope(
        const std::string& marker_name,
        const ObjectPointSet& rope,
        const std::vector<std_msgs::ColorRGBA>& colors,
        const int32_t id) const
{
    if (!disable_all_visualizations_)
    {
        visualization_msgs::Marker marker;

        marker.header.frame_id = world_frame_name_;

        marker.type = visualization_msgs::Marker::LINE_STRIP;
        marker.ns = marker_name;
        marker.id = id;
        marker.scale.x = 0.005;
        marker.points = EigenHelpersConversions::EigenMatrix3XdToVectorGeometryPoint(rope);
        marker.colors = colors;

        // Assumes that all non specified values are 0.0
        marker.pose.orientation.w = 1.0;

        marker.header.stamp = ros::Time::now();
//        publish(marker);

        marker.type = visualization_msgs::Marker::POINTS;
//        marker.type = visualization_msgs::Marker::SPHERE;
//        marker.id = id + 1;
//        const double scale = 0.015;
        const double scale = 0.005;
        marker.scale.x = scale;
        marker.scale.y = scale;
        marker.scale.z = scale;

        // Assumes that all non specified values are 0.0
        marker.pose.orientation.w = 1.0;

        marker.header.stamp = ros::Time::now();
        publish(marker);
    }
}

void Visualizer::visualizeCloth(
        const std::string& marker_name,
        const ObjectPointSet& cloth,
        const std_msgs::ColorRGBA& color,
        const int32_t id) const
{
    if (!disable_all_visualizations_)
    {
        const std::vector<std_msgs::ColorRGBA> colors((size_t)cloth.cols(), color);
        visualizeCloth(marker_name, cloth, colors, id);
    }
}

void Visualizer::visualizeCloth(
        const std::string& marker_name,
        const ObjectPointSet& cloth,
        const std::vector<std_msgs::ColorRGBA>& colors,
        const int32_t id) const
{
    if (!disable_all_visualizations_)
    {
        visualization_msgs::Marker marker;

        marker.header.frame_id = world_frame_name_;

        marker.type = visualization_msgs::Marker::POINTS;
        marker.ns = marker_name;
        marker.id = id;
        marker.scale.x = 0.005;
        marker.scale.y = 0.005;
        marker.points = EigenHelpersConversions::EigenMatrix3XdToVectorGeometryPoint(cloth);
        marker.colors = colors;

        // Assumes that all non specified values are 0.0
        marker.pose.orientation.w = 1.0;

        marker.header.stamp = ros::Time::now();
        publish(marker);
    }
}

void Visualizer::visualizeGripper(
        const std::string& marker_name,
        const Eigen::Isometry3d& eigen_pose,
        const std_msgs::ColorRGBA& color,
        const int32_t id) const
{
    if (!disable_all_visualizations_)
    {
        visualization_msgs::Marker marker;

        marker.header.frame_id = world_frame_name_;
        marker.header.stamp = ros::Time::now();

        marker.type = visualization_msgs::Marker::CUBE_LIST;
        marker.ns = marker_name;
        marker.id = id;
        marker.scale.x = 0.03;
        marker.scale.y = 0.03;
        marker.scale.z = 0.01;
        marker.pose = EigenHelpersConversions::EigenIsometry3dToGeometryPose(eigen_pose);
        marker.colors = {color, color};

        geometry_msgs::Point p;
        p.x = 0.0;
        p.y = 0.0;
        p.z = gripper_apperture_ * 0.5;
        marker.points.push_back(p);
        p.z = -gripper_apperture_ * 0.5;
        marker.points.push_back(p);

        publish(marker);
    }
}

void Visualizer::visualizeGrippers(
        const std::string& marker_name,
        const EigenHelpers::VectorIsometry3d eigen_poses,
        const std_msgs::ColorRGBA& color,
        const int32_t id) const
{
    if (!disable_all_visualizations_)
    {
        for (size_t gripper_ind = 0; gripper_ind < eigen_poses.size(); ++gripper_ind)
        {
            visualizeGripper(marker_name, eigen_poses[gripper_ind], color, id + (int32_t)gripper_ind);
        }
    }
}

void Visualizer::visualizeObjectDelta(
        const std::string& marker_name,
        const ObjectPointSet& current,
        const ObjectPointSet& desired,
        const std_msgs::ColorRGBA& color,
        const int32_t id) const
{
    if (!disable_all_visualizations_)
    {
        visualization_msgs::Marker marker;

        marker.header.frame_id = world_frame_name_;

        marker.type = visualization_msgs::Marker::LINE_LIST;
        marker.ns = marker_name;
        marker.id = id;
        marker.scale.x = 0.001;
        marker.points.reserve((size_t)current.cols() * 2);
        marker.colors.reserve((size_t)current.cols() * 2);
        for (ssize_t col = 0; col < current.cols(); col++)
        {
            marker.points.push_back(EigenHelpersConversions::EigenVector3dToGeometryPoint(current.col(col)));
            marker.points.push_back(EigenHelpersConversions::EigenVector3dToGeometryPoint(desired.col(col)));
            marker.colors.push_back(color);
            marker.colors.push_back(color);
        }

        // Assumes that all non specified values are 0.0
        marker.pose.orientation.w = 1.0;

        marker.header.stamp = ros::Time::now();
        publish(marker);
    }
}

void Visualizer::visualizeTranslation(
        const std::string& marker_name,
        const geometry_msgs::Point& start,
        const geometry_msgs::Point& end,
        const std_msgs::ColorRGBA& color,
        const int32_t id) const
{
    if (!disable_all_visualizations_)
    {
        visualization_msgs::Marker marker;

        marker.header.frame_id = world_frame_name_;

        marker.type = visualization_msgs::Marker::LINE_STRIP;
        marker.ns = marker_name;
        marker.id = id;
        marker.scale.x = 0.01;
        marker.points.push_back(start);
        marker.points.push_back(end);
        marker.colors.push_back(color);
        marker.colors.push_back(color);

        // Assumes that all non specified values are 0.0
        marker.pose.orientation.w = 1.0;

        marker.header.stamp = ros::Time::now();
        publish(marker);
    }
}

void Visualizer::visualizeTranslation(
        const std::string& marker_name,
        const Eigen::Vector3d& start,
        const Eigen::Vector3d& end,
        const std_msgs::ColorRGBA& color,
        const int32_t id) const
{
    if (!disable_all_visualizations_)
    {
        visualizeTranslation(
                    marker_name,
                    EigenHelpersConversions::EigenVector3dToGeometryPoint(start),
                    EigenHelpersConversions::EigenVector3dToGeometryPoint(end),
                    color,
                    id);
    }
}

void Visualizer::visualizeTranslation(
        const std::string& marker_name,
        const Eigen::Isometry3d &start,
        const Eigen::Isometry3d &end,
        const std_msgs::ColorRGBA& color,
        const int32_t id) const
{
    if (!disable_all_visualizations_)
    {
        Visualizer::visualizeTranslation(
                    marker_name,
                    start.translation(),
                    end.translation(),
                    color,
                    id);
    }
}

void Visualizer::visualizeLines(
        const std::string& marker_name,
        const EigenHelpers::VectorVector3d& start,
        const EigenHelpers::VectorVector3d& end,
        const std_msgs::ColorRGBA& color,
        const int32_t id,
        const double scale) const
{
    if (!disable_all_visualizations_)
    {
        assert(start.size() == end.size());

        visualization_msgs::Marker marker;

        marker.header.frame_id = world_frame_name_;

        marker.type = visualization_msgs::Marker::LINE_LIST;
        marker.ns = marker_name;
        marker.id = id;
        marker.scale.x = scale;

        for (size_t ind = 0; ind < start.size(); ind++)
        {
            marker.points.push_back(EigenHelpersConversions::EigenVector3dToGeometryPoint(start[ind]));
            marker.points.push_back(EigenHelpersConversions::EigenVector3dToGeometryPoint(end[ind]));
            marker.colors.push_back(color);
            marker.colors.push_back(color);
        }

        // Assumes that all non specified values are 0.0
        marker.pose.orientation.w = 1.0;

        marker.header.stamp = ros::Time::now();
        publish(marker);
    }
}

void Visualizer::visualizeLineStrip(
        const std::string& marker_name,
        const EigenHelpers::VectorVector3d& point_sequence,
        const std_msgs::ColorRGBA& color,
        const int32_t id,
        const double scale) const
{
    if (!disable_all_visualizations_)
    {
        visualization_msgs::Marker marker;

        marker.header.frame_id = world_frame_name_;

        marker.type = visualization_msgs::Marker::LINE_STRIP;
        marker.ns = marker_name;
        marker.id = id;
        marker.scale.x = scale;

        marker.points = EigenHelpersConversions::VectorEigenVector3dToVectorGeometryPoint(point_sequence);
        marker.colors = std::vector<std_msgs::ColorRGBA>(marker.points.size(), color);

        // Assumes that all non specified values are 0.0
        marker.pose.orientation.w = 1.0;

        marker.header.stamp = ros::Time::now();
        publish(marker);
    }
}

void Visualizer::visualizeXYZTrajectory(
        const std::string& marker_name,
        const EigenHelpers::VectorVector3d& point_sequence,
        const std_msgs::ColorRGBA& color,
        const int32_t id) const
{
    if (!disable_all_visualizations_)
    {
        visualizeLineStrip(marker_name, point_sequence, color, id);
    }
}


void Visualizer::publishAsyncMain()
{
    const double freq = ROSHelpers::GetParam<double>(ph_, "async_publish_frequency", 20.0);
    ros::Rate rate(freq);
    while (ros::ok())
    {
        {
            std::lock_guard<std::mutex> lock(markers_mtx_);
            visualization_maker_array_pub_.publish(async_markers_);
        }
        rate.sleep();
    }
}
