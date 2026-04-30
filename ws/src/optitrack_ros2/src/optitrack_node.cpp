//=============================================================================
// OptiTrack NatNet ROS2 Node
// Publishes rigid body poses from OptiTrack to ROS2 topics
//=============================================================================

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include <NatNetTypes.h>
#include <NatNetCAPI.h>
#include <NatNetClient.h>

#include <memory>
#include <map>
#include <string>
#include <mutex>

// Forward declarations
void NATNET_CALLCONV DataHandler(sFrameOfMocapData* data, void* pUserData);
void NATNET_CALLCONV MessageHandler(Verbosity msgType, const char* msg);

class OptiTrackNode : public rclcpp::Node
{
public:
    OptiTrackNode() : Node("optitrack_node")
    {
        // Declare parameters
        this->declare_parameter<std::string>("server_address", "10.10.30.3");
        this->declare_parameter<std::string>("local_address", "10.10.30.2");
        this->declare_parameter<int>("command_port", 1510);
        this->declare_parameter<int>("data_port", 1511);
        this->declare_parameter<int>("rigid_body_id", -1);
        this->declare_parameter<std::string>("world_frame", "world");

        // Tracking mode: "rigid_body", "marker" (single marker tracking), or "both"
        this->declare_parameter<std::string>("tracking_mode", "both");
        // Marker ID to track (-1 = track largest marker by size)
        this->declare_parameter<int>("marker_id", -1);

        // Get parameters
        server_address_ = this->get_parameter("server_address").as_string();
        local_address_ = this->get_parameter("local_address").as_string();
        command_port_ = this->get_parameter("command_port").as_int();
        data_port_ = this->get_parameter("data_port").as_int();
        rigid_body_id_ = this->get_parameter("rigid_body_id").as_int();
        world_frame_ = this->get_parameter("world_frame").as_string();
        tracking_mode_ = this->get_parameter("tracking_mode").as_string();
        marker_id_ = this->get_parameter("marker_id").as_int();

        // Print NatNet version
        unsigned char ver[4];
        NatNet_GetVersion(ver);
        RCLCPP_INFO(this->get_logger(), "NatNet Client Version: %d.%d.%d.%d", 
                    ver[0], ver[1], ver[2], ver[3]);

        // Install logging callback
        NatNet_SetLogCallback(MessageHandler);

        // Create NatNet client
        natnet_client_ = std::make_unique<NatNetClient>();
        
        // Set frame callback
        natnet_client_->SetFrameReceivedCallback(DataHandler, this);

        // Connect to server
        if (!connectToServer()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to connect to OptiTrack server");
            return;
        }

        // Get data descriptions
        if (!updateDataDescriptions()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to get data descriptions");
            return;
        }

        // Log tracking mode
        RCLCPP_INFO(this->get_logger(), "Tracking mode: %s", tracking_mode_.c_str());
        if (tracking_mode_ == "marker" || tracking_mode_ == "both") {
            if (marker_id_ == -1) {
                RCLCPP_INFO(this->get_logger(), "Will track largest unlabeled marker by size");
            } else {
                RCLCPP_INFO(this->get_logger(), "Will track marker ID: %d", marker_id_);
            }
            // Create marker publisher
            marker_publisher_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
                "optitrack/marker/pose", 10);
            RCLCPP_INFO(this->get_logger(), "Created marker publisher on topic: optitrack/marker/pose");
        }

        if (tracking_mode_ == "rigid_body" || tracking_mode_ == "both") {
            RCLCPP_INFO(this->get_logger(), "Rigid body tracking enabled");
        }

        RCLCPP_INFO(this->get_logger(), "OptiTrack ROS2 node initialized and ready");
    }

    ~OptiTrackNode()
    {
        if (natnet_client_) {
            natnet_client_->Disconnect();
        }
        if (data_defs_) {
            NatNet_FreeDescriptions(data_defs_);
        }
    }

    void processFrame(sFrameOfMocapData* data)
    {
        std::lock_guard<std::mutex> lock(publishers_mutex_);

        auto timestamp = this->now();

        // Process based on tracking mode
        if (tracking_mode_ == "marker" || tracking_mode_ == "both") {
            processMarkers(data, timestamp);
        }
        
        if (tracking_mode_ == "rigid_body" || tracking_mode_ == "both") {
            processRigidBodies(data, timestamp);
        }
    }
    
    void processMarkers(sFrameOfMocapData* data, const rclcpp::Time& timestamp)
    {
        // Debug: log number of markers periodically
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
            "Labeled markers: %d, Unlabeled markers: %d",
            data->nLabeledMarkers, data->nOtherMarkers);
        
        // Find the marker to track
        const sMarker* target_marker = nullptr;
        float largest_size = 0.0f;
        
        for (int i = 0; i < data->nLabeledMarkers; i++) {
            const sMarker& m = data->LabeledMarkers[i];
            
            // Check if marker is occluded (bit 0)
            bool occluded = (m.params & 0x01) != 0;
            // Check if marker is unlabeled (bit 4)
            bool unlabeled = (m.params & 0x10) != 0;
            
            // Debug: log marker info periodically
            RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                "Marker ID=%d, Pos=(%.3f,%.3f,%.3f), Size=%.4f, Occluded=%d, Unlabeled=%d, Params=0x%04x",
                m.ID, m.x, m.y, m.z, m.size, occluded, unlabeled, m.params);
            
            if (occluded) {
                continue;
            }

            // If tracking the "largest" marker (marker_id_ == -1), we only consider unlabeled ones
            // to avoid jumping to a marker that is part of a rigid body.
            if (marker_id_ == -1 && !unlabeled) {
                continue;
            }
            
            if (marker_id_ != -1) {
                // Track specific marker by ID
                if (m.ID == marker_id_) {
                    target_marker = &m;
                    break;
                }
            } else {
                // Track largest marker by size
                if (m.size > largest_size) {
                    largest_size = m.size;
                    target_marker = &m;
                }
            }
        }
        
        // Publish marker position if found
        if (target_marker != nullptr && marker_publisher_ != nullptr) {
            geometry_msgs::msg::PoseStamped pose_msg;
            pose_msg.header.stamp = timestamp;
            pose_msg.header.frame_id = world_frame_;
            
            pose_msg.pose.position.x = target_marker->x;
            pose_msg.pose.position.y = target_marker->y;
            pose_msg.pose.position.z = target_marker->z;
            
            // Identity quaternion - no orientation info from single marker
            pose_msg.pose.orientation.x = 0.0;
            pose_msg.pose.orientation.y = 0.0;
            pose_msg.pose.orientation.z = 0.0;
            pose_msg.pose.orientation.w = 1.0;
            
            marker_publisher_->publish(pose_msg);
            
            RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 500,
                "Published marker ID=%d, Pos=(%.3f,%.3f,%.3f), Size=%.4f",
                target_marker->ID, target_marker->x, target_marker->y, target_marker->z, 
                target_marker->size);
        }
    }
    
    void processRigidBodies(sFrameOfMocapData* data, const rclcpp::Time& timestamp)
    {
        // Process rigid bodies
        for (int i = 0; i < data->nRigidBodies; i++) {
            const sRigidBodyData& rb = data->RigidBodies[i];
            
            // Check if tracking is valid
            bool tracking_valid = (rb.params & 0x01) != 0;

            // Debug output
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                "RigidBody ID=%d, Valid=%d, Pos=(%.3f,%.3f,%.3f), MeanError=%.3f",
                rb.ID, tracking_valid, rb.x, rb.y, rb.z, rb.MeanError);
            

            if (!tracking_valid) {
                continue;
            }

            int rb_id = rb.ID;

            // If rigid_body_id is specified, only publish that specific rigid body
            if (rigid_body_id_ != -1 && rb_id != rigid_body_id_) {
                continue;
            }


            // Create publisher if it doesn't exist
            if (pose_publishers_.find(rb_id) == pose_publishers_.end()) {
                std::string rb_name = getRigidBodyName(rb_id);
                std::string topic_name = "optitrack/" + rb_name + "/pose";
                pose_publishers_[rb_id] = this->create_publisher<geometry_msgs::msg::PoseStamped>(
                    topic_name, 10);
                RCLCPP_INFO(this->get_logger(), "Created publisher for %s (ID: %d) on topic %s", 
                           rb_name.c_str(), rb_id, topic_name.c_str());
            }

            // Create and publish PoseStamped message
            geometry_msgs::msg::PoseStamped pose_msg;
            pose_msg.header.stamp = timestamp;
            pose_msg.header.frame_id = world_frame_;
            
            pose_msg.pose.position.x = rb.x;
            pose_msg.pose.position.y = rb.y;
            pose_msg.pose.position.z = rb.z;
            
            pose_msg.pose.orientation.x = rb.qx;
            pose_msg.pose.orientation.y = rb.qy;
            pose_msg.pose.orientation.z = rb.qz;
            pose_msg.pose.orientation.w = rb.qw;

            pose_publishers_[rb_id]->publish(pose_msg);
        }
    }

private:
    bool connectToServer()
    {
        sNatNetClientConnectParams connect_params;
        // Use Unicast since Motive is broadcasting to 255.255.255.255
        connect_params.connectionType = ConnectionType_Unicast;
        connect_params.serverAddress = server_address_.c_str();
        connect_params.localAddress = local_address_.c_str();
        connect_params.serverCommandPort = command_port_;
        connect_params.serverDataPort = data_port_;
        connect_params.multicastAddress = nullptr;

        int ret = natnet_client_->Connect(connect_params);
        if (ret != ErrorCode_OK) {
            RCLCPP_ERROR(this->get_logger(), "Failed to connect. Error code: %d", ret);
            return false;
        }

        // Get server description
        sServerDescription server_desc;
        ret = natnet_client_->GetServerDescription(&server_desc);
        if (ret != ErrorCode_OK || !server_desc.HostPresent) {
            RCLCPP_ERROR(this->get_logger(), "Server not present");
            return false;
        }

        RCLCPP_INFO(this->get_logger(), "Connected to %s (v%d.%d.%d.%d)",
                   server_desc.szHostApp,
                   server_desc.HostAppVersion[0],
                   server_desc.HostAppVersion[1],
                   server_desc.HostAppVersion[2],
                   server_desc.HostAppVersion[3]);

        return true;
    }

    bool updateDataDescriptions()
    {
        if (data_defs_) {
            NatNet_FreeDescriptions(data_defs_);
            data_defs_ = nullptr;
        }

        int ret = natnet_client_->GetDataDescriptionList(&data_defs_);
        if (ret != ErrorCode_OK || data_defs_ == nullptr) {
            return false;
        }

        // Build rigid body name map
        rigid_body_names_.clear();
        for (int i = 0; i < data_defs_->nDataDescriptions; i++) {
            if (data_defs_->arrDataDescriptions[i].type == Descriptor_RigidBody) {
                sRigidBodyDescription* rb_desc = 
                    data_defs_->arrDataDescriptions[i].Data.RigidBodyDescription;
                rigid_body_names_[rb_desc->ID] = std::string(rb_desc->szName);
                RCLCPP_INFO(this->get_logger(), "Found rigid body: %s (ID: %d)",
                           rb_desc->szName, rb_desc->ID);
            }
        }

        // If rigid_body_id is specified, verify it exists
        if (rigid_body_id_ != -1) {
            if (rigid_body_names_.find(rigid_body_id_) == rigid_body_names_.end()) {
                RCLCPP_WARN(this->get_logger(), 
                           "Specified rigid_body_id %d not found in available rigid bodies",
                           rigid_body_id_);
            } else {
                RCLCPP_INFO(this->get_logger(), "Tracking rigid body ID %d: %s",
                           rigid_body_id_, rigid_body_names_[rigid_body_id_].c_str());
            }
        } else {
            RCLCPP_INFO(this->get_logger(), "Tracking all rigid bodies (rigid_body_id = -1)");
        }

        return true;
    }

    std::string getRigidBodyName(int id)
    {
        auto it = rigid_body_names_.find(id);
        if (it != rigid_body_names_.end()) {
            return it->second;
        }
        return "rigid_body_" + std::to_string(id);
    }

    // ROS2 members
    std::map<int, rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr> pose_publishers_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr marker_publisher_;
    std::mutex publishers_mutex_;

    // NatNet members
    std::unique_ptr<NatNetClient> natnet_client_;
    sDataDescriptions* data_defs_ = nullptr;
    std::map<int, std::string> rigid_body_names_;

    // Parameters
    std::string server_address_;
    std::string local_address_;
    int command_port_;
    int data_port_;
    int rigid_body_id_;
    std::string world_frame_;
    std::string tracking_mode_;
    int marker_id_;
};

// Global pointer for callbacks
OptiTrackNode* g_node_ptr = nullptr;

void NATNET_CALLCONV DataHandler(sFrameOfMocapData* data, void* pUserData)
{
    OptiTrackNode* node = static_cast<OptiTrackNode*>(pUserData);
    if (node && data) {
        node->processFrame(data);
    }
}

void NATNET_CALLCONV MessageHandler(Verbosity msgType, const char* msg)
{
    if (g_node_ptr == nullptr) return;

    switch (msgType) {
        case Verbosity_Debug:
            RCLCPP_DEBUG(g_node_ptr->get_logger(), "[NatNet] %s", msg);
            break;
        case Verbosity_Info:
            RCLCPP_INFO(g_node_ptr->get_logger(), "[NatNet] %s", msg);
            break;
        case Verbosity_Warning:
            RCLCPP_WARN(g_node_ptr->get_logger(), "[NatNet] %s", msg);
            break;
        case Verbosity_Error:
            RCLCPP_ERROR(g_node_ptr->get_logger(), "[NatNet] %s", msg);
            break;
        default:
            break;
    }
}

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<OptiTrackNode>();
    g_node_ptr = node.get();
    
    rclcpp::spin(node);
    
    g_node_ptr = nullptr;
    rclcpp::shutdown();
    return 0;
}
