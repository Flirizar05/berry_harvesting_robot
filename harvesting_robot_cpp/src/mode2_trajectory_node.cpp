#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>

#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/string.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include <kdl/chain.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/frames.hpp>
#include <kdl/jntarray.hpp>
#include <kdl/tree.hpp>
#include <kdl_parser/kdl_parser.hpp>

#include "ompl/base/Goal.h"
#include "ompl/base/PlannerData.h"
#include "ompl/base/PlannerTerminationCondition.h"
#include "ompl/base/ProblemDefinition.h"
#include "ompl/base/ScopedState.h"
#include "ompl/base/spaces/HybridStateSpace.h"
#include "ompl/base/spaces/RealVectorStateSpace.h"
#include "ompl/control/ODESolver.h"
#include "ompl/control/PathControl.h"
#include "ompl/control/SpaceInformation.h"
#include "ompl/control/planners/rrt/HyRRT.h"
#include "ompl/control/spaces/RealVectorControlSpace.h"

namespace ob = ompl::base;
namespace oc = ompl::control;

// Shared workspace and goal used by the HyRRT callbacks.
static std::array<double, 3> g_workspace_min = {-0.6, -0.4, 0.10};
static std::array<double, 3> g_workspace_max = {0.2, 0.4, 0.70};
static std::array<double, 3> g_goal_xyz = {0.30, 0.0, 0.30};
static double g_jump_radius = 0.0;

static std::string getDefaultUrdfPath()
{
  const std::string package_share =
    ament_index_cpp::get_package_share_directory("harvesting_robot");
  return package_share + "/urdf/elfin3.urdf";
}

bool flowSet(oc::HyRRT::Motion *motion)
{
  auto *values = motion->state
                   ->as<ob::HybridStateSpace::StateType>()
                   ->as<ob::RealVectorStateSpace::StateType>(0)
                   ->values;

  const double x = values[0];
  const double y = values[1];
  const double z = values[2];

  return (x >= g_workspace_min[0] && x <= g_workspace_max[0]) &&
         (y >= g_workspace_min[1] && y <= g_workspace_max[1]) &&
         (z >= g_workspace_min[2] && z <= g_workspace_max[2]);
}

bool jumpSet(oc::HyRRT::Motion *motion)
{
  if (g_jump_radius <= 0.0) {
    return false;
  }

  auto *values = motion->state
                   ->as<ob::HybridStateSpace::StateType>()
                   ->as<ob::RealVectorStateSpace::StateType>(0)
                   ->values;

  const double dx = values[0] - g_goal_xyz[0];
  const double dy = values[1] - g_goal_xyz[1];
  const double dz = values[2] - g_goal_xyz[2];
  const double distance = std::sqrt(dx * dx + dy * dy + dz * dz);

  return distance <= g_jump_radius;
}

bool unsafeSet(oc::HyRRT::Motion *motion)
{
  (void)motion;
  return false;
}

void flowODE(
  const oc::ODESolver::StateType &current_state,
  const oc::Control *control,
  oc::ODESolver::StateType &state_derivative)
{
  (void)current_state;

  const double *control_values =
    control->as<oc::RealVectorControlSpace::ControlType>()->values;

  state_derivative.resize(3, 0.0);
  state_derivative[0] = control_values[0];
  state_derivative[1] = control_values[1];
  state_derivative[2] = control_values[2];
}

ob::State *discreteSimulator(
  ob::State *current_state,
  const oc::Control *control,
  ob::State *new_state)
{
  (void)control;

  auto *source = current_state
                   ->as<ob::HybridStateSpace::StateType>()
                   ->as<ob::RealVectorStateSpace::StateType>(0);
  auto *destination = new_state
                        ->as<ob::HybridStateSpace::StateType>()
                        ->as<ob::RealVectorStateSpace::StateType>(0);

  for (unsigned int index = 0; index < 3; ++index) {
    destination->values[index] = source->values[index];
  }

  return new_state;
}

class EEGoalRegion : public ob::Goal
{
public:
  EEGoalRegion(
    const ob::SpaceInformationPtr &space_information,
    const std::array<double, 3> &goal_xyz,
    double tolerance)
  : ob::Goal(space_information), goal_xyz_(goal_xyz), tolerance_(tolerance)
  {
  }

  bool isSatisfied(const ob::State *state, double *distance) const override
  {
    auto *values = state
                     ->as<ob::HybridStateSpace::StateType>()
                     ->as<ob::RealVectorStateSpace::StateType>(0)
                     ->values;

    const double dx = values[0] - goal_xyz_[0];
    const double dy = values[1] - goal_xyz_[1];
    const double dz = values[2] - goal_xyz_[2];
    const double norm = std::sqrt(dx * dx + dy * dy + dz * dz);

    if (distance != nullptr) {
      *distance = norm;
    }
    return norm <= tolerance_;
  }

  bool isSatisfied(const ob::State *state) const override
  {
    double distance = 0.0;
    return isSatisfied(state, &distance);
  }

private:
  std::array<double, 3> goal_xyz_;
  double tolerance_;
};

static bool fileExistsAndIsNonEmpty(const std::string &path)
{
  std::ifstream file(path, std::ios::binary);
  if (!file.good()) {
    return false;
  }

  file.seekg(0, std::ios::end);
  const auto size = file.tellg();
  return size > 10;
}

static std::string toUpper(std::string text)
{
  for (auto &character : text) {
    character = static_cast<char>(::toupper(character));
  }
  return text;
}

static bool isFinitePoint(const std::array<double, 3> &point)
{
  return std::isfinite(point[0]) &&
         std::isfinite(point[1]) &&
         std::isfinite(point[2]);
}

static bool isInsideWorkspace(const std::array<double, 3> &point)
{
  return (point[0] >= g_workspace_min[0] && point[0] <= g_workspace_max[0]) &&
         (point[1] >= g_workspace_min[1] && point[1] <= g_workspace_max[1]) &&
         (point[2] >= g_workspace_min[2] && point[2] <= g_workspace_max[2]);
}

static std::array<double, 3> clampToWorkspace(const std::array<double, 3> &point)
{
  return {
    std::min(std::max(point[0], g_workspace_min[0]), g_workspace_max[0]),
    std::min(std::max(point[1], g_workspace_min[1]), g_workspace_max[1]),
    std::min(std::max(point[2], g_workspace_min[2]), g_workspace_max[2])};
}

class Mode2TrajectoryNode : public rclcpp::Node
{
public:
  Mode2TrajectoryNode()
  : Node("mode2_trajectory_node")
  {
    declareParameters();
    loadParameters();
    createRosInterfaces();
    kdl_ready_ = initializeKdl();
    buildOmpl();

    waypoint_timer_ = this->create_wall_timer(
      std::chrono::duration<double>(waypoint_publish_period_sec_),
      std::bind(&Mode2TrajectoryNode::publishWaypointLoop, this));

    republish_timer_ = this->create_wall_timer(
      std::chrono::seconds(1),
      std::bind(&Mode2TrajectoryNode::republishLatchedArtifacts, this));

    RCLCPP_INFO(
      this->get_logger(),
      "mode2_trajectory_node ready. "
      "ws=[(%.3f,%.3f)x(%.3f,%.3f)x(%.3f,%.3f)] "
      "wp_tol=%.3f goal_tol=%.3f plan_t=%.2f vmax=%.2f step=%.3f clamp=%s",
      g_workspace_min[0], g_workspace_max[0],
      g_workspace_min[1], g_workspace_max[1],
      g_workspace_min[2], g_workspace_max[2],
      waypoint_tolerance_m_,
      goal_tolerance_m_,
      planning_time_sec_,
      max_cartesian_velocity_,
      flow_step_sec_,
      clamp_to_workspace_ ? "true" : "false");
  }

private:
  void declareParameters()
  {
    this->declare_parameter<std::string>("base_frame", "elfin_base");
    this->declare_parameter<std::string>("target_topic", "/target_base");
    this->declare_parameter<std::string>("joint_state_topic", "/joint_states");

    this->declare_parameter<std::string>("cmd_topic", "/trajectory/cmd");
    this->declare_parameter<std::string>("status_topic", "/trajectory/status");
    this->declare_parameter<std::string>("waypoint_topic", "/trajectory/waypoint");
    this->declare_parameter<std::string>("path_topic", "/trajectory/path");

    this->declare_parameter<std::string>("tree_marker_topic", "/trajectory/hyrrt_tree_marker");
    this->declare_parameter<std::string>("path_marker_topic", "/trajectory/hyrrt_path_marker");
    this->declare_parameter<std::string>("start_marker_topic", "/trajectory/start_marker");
    this->declare_parameter<std::string>("goal_marker_topic", "/trajectory/goal_marker");
    this->declare_parameter<std::string>("waypoint_marker_topic", "/trajectory/waypoint_marker");
    this->declare_parameter<std::string>("tcp_marker_topic", "/trajectory/tcp_marker");

    this->declare_parameter<std::vector<double>>("ws_min", {-0.95, -0.9, 0.05});
    this->declare_parameter<std::vector<double>>("ws_max", {0.40, 0.9, 0.90});

    this->declare_parameter<double>("planning_time", 5.0);
    this->declare_parameter<double>("max_cartesian_vel", 0.15);
    this->declare_parameter<double>("flow_step", 0.02);
    this->declare_parameter<double>("Tm_min", 0.50);
    this->declare_parameter<double>("D_radius", 0.0);

    this->declare_parameter<double>("waypoint_tol_m", 0.05);
    this->declare_parameter<double>("goal_tol_m", 0.05);
    this->declare_parameter<double>("waypoint_dt", 0.02);

    this->declare_parameter<std::string>("urdf_path", getDefaultUrdfPath());
    this->declare_parameter<std::string>("ee_link", "rg2ft_grasp_point");
    this->declare_parameter<std::vector<std::string>>(
      "joint_names",
      {"elfin_joint1", "elfin_joint2", "elfin_joint3", "elfin_joint4", "elfin_joint5", "elfin_joint6"});

    this->declare_parameter<bool>("clamp_to_workspace", false);
  }

  void loadParameters()
  {
    base_frame_ = this->get_parameter("base_frame").as_string();
    target_topic_ = this->get_parameter("target_topic").as_string();
    joint_state_topic_ = this->get_parameter("joint_state_topic").as_string();

    command_topic_ = this->get_parameter("cmd_topic").as_string();
    status_topic_ = this->get_parameter("status_topic").as_string();
    waypoint_topic_ = this->get_parameter("waypoint_topic").as_string();
    path_topic_ = this->get_parameter("path_topic").as_string();

    tree_marker_topic_ = this->get_parameter("tree_marker_topic").as_string();
    path_marker_topic_ = this->get_parameter("path_marker_topic").as_string();
    start_marker_topic_ = this->get_parameter("start_marker_topic").as_string();
    goal_marker_topic_ = this->get_parameter("goal_marker_topic").as_string();
    waypoint_marker_topic_ = this->get_parameter("waypoint_marker_topic").as_string();
    tcp_marker_topic_ = this->get_parameter("tcp_marker_topic").as_string();

    const auto workspace_min = this->get_parameter("ws_min").as_double_array();
    const auto workspace_max = this->get_parameter("ws_max").as_double_array();
    for (int index = 0; index < 3; ++index) {
      g_workspace_min[index] = workspace_min[index];
      g_workspace_max[index] = workspace_max[index];
    }

    planning_time_sec_ = this->get_parameter("planning_time").as_double();
    max_cartesian_velocity_ = this->get_parameter("max_cartesian_vel").as_double();
    flow_step_sec_ = this->get_parameter("flow_step").as_double();
    minimum_tm_sec_ = this->get_parameter("Tm_min").as_double();
    g_jump_radius = this->get_parameter("D_radius").as_double();

    waypoint_tolerance_m_ = this->get_parameter("waypoint_tol_m").as_double();
    goal_tolerance_m_ = this->get_parameter("goal_tol_m").as_double();
    waypoint_publish_period_sec_ = this->get_parameter("waypoint_dt").as_double();

    urdf_path_ = this->get_parameter("urdf_path").as_string();
    ee_link_ = this->get_parameter("ee_link").as_string();
    joint_names_ = this->get_parameter("joint_names").as_string_array();

    clamp_to_workspace_ = this->get_parameter("clamp_to_workspace").as_bool();
  }

  void createRosInterfaces()
  {
    status_publisher_ =
      this->create_publisher<std_msgs::msg::String>(status_topic_, 10);
    publishStatus("IDLE");

    waypoint_publisher_ =
      this->create_publisher<geometry_msgs::msg::PoseStamped>(waypoint_topic_, 10);

    rclcpp::QoS latched_qos(1);
    latched_qos.transient_local().reliable();

    path_publisher_ =
      this->create_publisher<nav_msgs::msg::Path>(path_topic_, latched_qos);

    tree_marker_publisher_ =
      this->create_publisher<visualization_msgs::msg::Marker>(tree_marker_topic_, latched_qos);
    path_marker_publisher_ =
      this->create_publisher<visualization_msgs::msg::Marker>(path_marker_topic_, latched_qos);
    start_marker_publisher_ =
      this->create_publisher<visualization_msgs::msg::Marker>(start_marker_topic_, latched_qos);
    goal_marker_publisher_ =
      this->create_publisher<visualization_msgs::msg::Marker>(goal_marker_topic_, latched_qos);
    waypoint_marker_publisher_ =
      this->create_publisher<visualization_msgs::msg::Marker>(waypoint_marker_topic_, 10);
    tcp_marker_publisher_ =
      this->create_publisher<visualization_msgs::msg::Marker>(tcp_marker_topic_, 10);

    target_subscription_ =
      this->create_subscription<geometry_msgs::msg::PointStamped>(
      target_topic_,
      10,
      std::bind(&Mode2TrajectoryNode::onTarget, this, std::placeholders::_1));

    joint_state_subscription_ =
      this->create_subscription<sensor_msgs::msg::JointState>(
      joint_state_topic_,
      50,
      std::bind(&Mode2TrajectoryNode::onJointState, this, std::placeholders::_1));

    command_subscription_ =
      this->create_subscription<std_msgs::msg::String>(
      command_topic_,
      10,
      std::bind(&Mode2TrajectoryNode::onCommand, this, std::placeholders::_1));
  }

  void publishStatus(const std::string &status_text)
  {
    std_msgs::msg::String status_msg;
    status_msg.data = status_text;
    status_publisher_->publish(status_msg);
  }

  bool initializeKdl()
  {
    if (!fileExistsAndIsNonEmpty(urdf_path_)) {
      RCLCPP_ERROR(this->get_logger(), "URDF missing or empty: %s", urdf_path_.c_str());
      return false;
    }

    KDL::Tree tree;
    if (!kdl_parser::treeFromFile(urdf_path_, tree)) {
      RCLCPP_ERROR(this->get_logger(), "Failed to parse URDF: %s", urdf_path_.c_str());
      return false;
    }

    if (!tree.getChain(base_frame_, ee_link_, kdl_chain_)) {
      RCLCPP_ERROR(
        this->get_logger(),
        "Failed to build KDL chain %s -> %s",
        base_frame_.c_str(),
        ee_link_.c_str());
      return false;
    }

    fk_solver_ = std::make_unique<KDL::ChainFkSolverPos_recursive>(kdl_chain_);

    RCLCPP_INFO(
      this->get_logger(),
      "KDL ready. chain_joints=%u | urdf=%s",
      kdl_chain_.getNrOfJoints(),
      urdf_path_.c_str());

    return true;
  }

  bool computeForwardKinematics(
    const std::vector<double> &joint_positions,
    std::array<double, 3> &tcp_xyz)
  {
    if (!kdl_ready_) {
      return false;
    }

    if (joint_positions.size() != kdl_chain_.getNrOfJoints()) {
      return false;
    }

    KDL::JntArray kdl_joint_positions(kdl_chain_.getNrOfJoints());
    for (unsigned int index = 0; index < kdl_chain_.getNrOfJoints(); ++index) {
      kdl_joint_positions(index) = joint_positions[index];
    }

    KDL::Frame tcp_frame;
    const int result = fk_solver_->JntToCart(kdl_joint_positions, tcp_frame);
    if (result < 0) {
      return false;
    }

    tcp_xyz[0] = tcp_frame.p.x();
    tcp_xyz[1] = tcp_frame.p.y();
    tcp_xyz[2] = tcp_frame.p.z();
    return true;
  }

  void buildOmpl()
  {
    auto *real_vector_space = new ob::RealVectorStateSpace(0);
    real_vector_space->addDimension(g_workspace_min[0], g_workspace_max[0]);
    real_vector_space->addDimension(g_workspace_min[1], g_workspace_max[1]);
    real_vector_space->addDimension(g_workspace_min[2], g_workspace_max[2]);
    ob::StateSpacePtr state_space(real_vector_space);

    auto *hybrid_space = new ob::HybridStateSpace(state_space);
    hybrid_state_space_ = ob::StateSpacePtr(hybrid_space);

    auto *flow_control_space = new oc::RealVectorControlSpace(hybrid_state_space_, 3);
    auto *jump_control_space = new oc::RealVectorControlSpace(hybrid_state_space_, 3);

    ob::RealVectorBounds flow_bounds(3);
    for (int index = 0; index < 3; ++index) {
      flow_bounds.setLow(index, -max_cartesian_velocity_);
      flow_bounds.setHigh(index, max_cartesian_velocity_);
    }
    flow_control_space->setBounds(flow_bounds);

    ob::RealVectorBounds jump_bounds(3);
    for (int index = 0; index < 3; ++index) {
      jump_bounds.setLow(index, 0.0);
      jump_bounds.setHigh(index, 0.0);
    }
    jump_control_space->setBounds(jump_bounds);

    oc::ControlSpacePtr flow_control_space_ptr(flow_control_space);
    oc::ControlSpacePtr jump_control_space_ptr(jump_control_space);

    auto *compound_control_space = new oc::CompoundControlSpace(hybrid_state_space_);
    compound_control_space->addSubspace(flow_control_space_ptr);
    compound_control_space->addSubspace(jump_control_space_ptr);
    oc::ControlSpacePtr compound_control_space_ptr(compound_control_space);

    space_information_ =
      std::make_shared<oc::SpaceInformation>(hybrid_state_space_, compound_control_space_ptr);

    oc::ODESolverPtr ode_solver(new oc::ODEBasicSolver<>(space_information_, &flowODE));
    space_information_->setStatePropagator(oc::ODESolver::getStatePropagator(ode_solver));
    space_information_->setPropagationStepSize(flow_step_sec_);
    space_information_->setup();

    problem_definition_ = std::make_shared<ob::ProblemDefinition>(space_information_);
    planner_ = std::make_shared<oc::HyRRT>(space_information_);
    planner_->setProblemDefinition(problem_definition_);
    planner_->setDiscreteSimulator(discreteSimulator);
    planner_->setFlowSet(flowSet);
    planner_->setJumpSet(jumpSet);
    planner_->setUnsafeSet(unsafeSet);

    const double tm = std::max(minimum_tm_sec_, flow_step_sec_);
    planner_->setFlowStepDuration(flow_step_sec_);
    planner_->setTm(tm);
    planner_->setup();

    RCLCPP_INFO(
      this->get_logger(),
      "OMPL ready. flow_step=%.4f Tm=%.4f vmax=%.3f",
      flow_step_sec_,
      tm,
      max_cartesian_velocity_);
  }

  void onTarget(const geometry_msgs::msg::PointStamped::SharedPtr msg)
  {
    if (!msg) {
      return;
    }

    std::array<double, 3> goal = {msg->point.x, msg->point.y, msg->point.z};
    if (!isFinitePoint(goal)) {
      return;
    }

    if (clamp_to_workspace_) {
      const auto clamped_goal = clampToWorkspace(goal);
      if (clamped_goal != goal) {
        RCLCPP_WARN(
          this->get_logger(),
          "Goal out of bounds -> clamped. raw=(%.3f,%.3f,%.3f) clamped=(%.3f,%.3f,%.3f)",
          goal[0], goal[1], goal[2],
          clamped_goal[0], clamped_goal[1], clamped_goal[2]);
      }
      goal = clamped_goal;
    }

    std::lock_guard<std::mutex> lock(goal_mutex_);
    goal_xyz_ = goal;
    g_goal_xyz = goal_xyz_;
    has_goal_ = true;
  }

  void onJointState(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    if (!msg) {
      return;
    }

    std::map<std::string, double> positions_by_joint;
    for (size_t index = 0; index < msg->name.size() && index < msg->position.size(); ++index) {
      positions_by_joint[msg->name[index]] = msg->position[index];
    }

    std::vector<double> joint_positions(joint_names_.size());
    for (size_t index = 0; index < joint_names_.size(); ++index) {
      const auto it = positions_by_joint.find(joint_names_[index]);
      if (it == positions_by_joint.end()) {
        return;
      }
      joint_positions[index] = it->second;
    }
    current_joint_positions_ = joint_positions;

    std::array<double, 3> tcp_xyz;
    if (computeForwardKinematics(current_joint_positions_, tcp_xyz) && isFinitePoint(tcp_xyz)) {
      tcp_xyz_ = tcp_xyz;
      has_forward_kinematics_ = true;
    }
  }

  void onCommand(const std_msgs::msg::String::SharedPtr msg)
  {
    if (!msg) {
      return;
    }

    const std::string command = toUpper(msg->data);

    if (command == "RESET") {
      is_active_ = false;
      is_planned_ = false;
      waypoints_.clear();
      waypoint_index_ = 0;
      publishStatus("IDLE");
      return;
    }

    if (command == "STOP" || command == "ABORT") {
      is_active_ = false;
      publishStatus("DONE_FAIL");
      publishStatus("IDLE");
      return;
    }

    if (command != "PLAN" && command != "START") {
      return;
    }

    if (is_busy_) {
      return;
    }

    publishStatus("BUSY");

    if (!kdl_ready_) {
      publishStatus("DONE_FAIL");
      publishStatus("IDLE");
      return;
    }

    if (!has_forward_kinematics_) {
      RCLCPP_WARN(this->get_logger(), "No FK available yet. Waiting for /joint_states.");
      publishStatus("DONE_FAIL");
      publishStatus("IDLE");
      return;
    }

    if (!has_goal_) {
      RCLCPP_WARN(this->get_logger(), "No goal available yet. Waiting for /target_base.");
      publishStatus("DONE_FAIL");
      publishStatus("IDLE");
      return;
    }

    if (command == "START" && is_planned_ && !waypoints_.empty()) {
      is_active_ = true;
      publishStatus("DONE_OK");
      publishStatus("IDLE");
      return;
    }

    is_busy_ = true;
    bool planned_ok = false;

    try {
      planned_ok = planOnce();
    } catch (const std::exception &exception) {
      RCLCPP_ERROR(this->get_logger(), "OMPL exception: %s", exception.what());
      planned_ok = false;
    } catch (...) {
      RCLCPP_ERROR(this->get_logger(), "Unknown exception during planning.");
      planned_ok = false;
    }

    is_busy_ = false;

    publishStatus(planned_ok ? "DONE_OK" : "DONE_FAIL");
    publishStatus("IDLE");

    if (planned_ok) {
      is_active_ = true;
    }
  }

  bool planOnce()
  {
    waypoints_.clear();
    waypoint_index_ = 0;
    is_planned_ = false;

    std::array<double, 3> start = tcp_xyz_;

    std::array<double, 3> goal;
    {
      std::lock_guard<std::mutex> lock(goal_mutex_);
      goal = goal_xyz_;
      g_goal_xyz = goal_xyz_;
    }

    if (!isFinitePoint(start) || !isFinitePoint(goal)) {
      return false;
    }

    if (!clamp_to_workspace_) {
      if (!isInsideWorkspace(start)) {
        RCLCPP_ERROR(
          this->get_logger(),
          "Start out of workspace. start=(%.3f,%.3f,%.3f) ws_z=[%.3f,%.3f]. "
          "Fix ws_min/ws_max to include tcp.",
          start[0], start[1], start[2], g_workspace_min[2], g_workspace_max[2]);
        return false;
      }

      if (!isInsideWorkspace(goal)) {
        RCLCPP_ERROR(
          this->get_logger(),
          "Goal out of workspace. goal=(%.3f,%.3f,%.3f). Fix ws_min/ws_max to include target.",
          goal[0], goal[1], goal[2]);
        return false;
      }
    } else {
      const auto clamped_start = clampToWorkspace(start);
      const auto clamped_goal = clampToWorkspace(goal);

      if (clamped_start != start) {
        RCLCPP_WARN(
          this->get_logger(),
          "Start clamped. raw=(%.3f,%.3f,%.3f) -> (%.3f,%.3f,%.3f)",
          start[0], start[1], start[2],
          clamped_start[0], clamped_start[1], clamped_start[2]);
        start = clamped_start;
      }

      if (clamped_goal != goal) {
        RCLCPP_WARN(
          this->get_logger(),
          "Goal clamped. raw=(%.3f,%.3f,%.3f) -> (%.3f,%.3f,%.3f)",
          goal[0], goal[1], goal[2],
          clamped_goal[0], clamped_goal[1], clamped_goal[2]);
        goal = clamped_goal;
      }

      if (!isInsideWorkspace(start) || !isInsideWorkspace(goal)) {
        return false;
      }
    }

    planner_->clear();
    problem_definition_->clearSolutionPaths();
    problem_definition_->clearStartStates();

    ob::ScopedState<> start_state(hybrid_state_space_);
    auto *start_values = start_state
                           ->as<ob::HybridStateSpace::StateType>()
                           ->as<ob::RealVectorStateSpace::StateType>(0)
                           ->values;
    start_values[0] = start[0];
    start_values[1] = start[1];
    start_values[2] = start[2];
    problem_definition_->addStartState(start_state);

    auto goal_region =
      std::make_shared<EEGoalRegion>(space_information_, goal, goal_tolerance_m_);
    problem_definition_->setGoal(goal_region);

    auto termination_condition =
      ob::timedPlannerTerminationCondition(planning_time_sec_);
    const ob::PlannerStatus solved = planner_->solve(termination_condition);

    if (!solved) {
      RCLCPP_WARN(this->get_logger(), "HyRRT: no solution in %.2f s", planning_time_sec_);
      is_planned_ = true;
      return false;
    }

    auto path = problem_definition_->getSolutionPath()->as<oc::PathControl>();
    const auto state_count = path->getStateCount();
    if (state_count < 2) {
      is_planned_ = true;
      return false;
    }

    nav_msgs::msg::Path path_msg;
    path_msg.header.frame_id = base_frame_;
    path_msg.header.stamp = this->now();

    for (unsigned int index = 0; index < state_count; ++index) {
      const ob::State *state = path->getState(index);
      auto *values = state
                       ->as<ob::HybridStateSpace::StateType>()
                       ->as<ob::RealVectorStateSpace::StateType>(0)
                       ->values;

      geometry_msgs::msg::Pose pose;
      pose.position.x = values[0];
      pose.position.y = values[1];
      pose.position.z = values[2];
      pose.orientation.w = 1.0;

      waypoints_.push_back(pose);

      geometry_msgs::msg::PoseStamped pose_stamped;
      pose_stamped.header = path_msg.header;
      pose_stamped.pose = pose;
      path_msg.poses.push_back(pose_stamped);
    }

    path_publisher_->publish(path_msg);
    last_path_msg_ = path_msg;

    publishStartAndGoalMarkers(start, goal);
    publishPathMarker(path_msg);
    publishTreeMarker();

    is_planned_ = true;

    RCLCPP_INFO(
      this->get_logger(),
      "HyRRT OK: %zu waypoints | start=(%.3f,%.3f,%.3f) goal=(%.3f,%.3f,%.3f) "
      "wp_tol=%.3f goal_tol=%.3f",
      waypoints_.size(),
      start[0], start[1], start[2],
      goal[0], goal[1], goal[2],
      waypoint_tolerance_m_,
      goal_tolerance_m_);

    return true;
  }

  void publishWaypointLoop()
  {
    if (!is_active_ || !is_planned_ || waypoints_.empty() || !has_forward_kinematics_) {
      return;
    }

    std::array<double, 3> tcp_xyz;
    if (!computeForwardKinematics(current_joint_positions_, tcp_xyz) || !isFinitePoint(tcp_xyz)) {
      return;
    }

    publishTcpMarker(tcp_xyz);

    const auto &current_waypoint = waypoints_[waypoint_index_];

    geometry_msgs::msg::PoseStamped waypoint_msg;
    waypoint_msg.header.frame_id = base_frame_;
    waypoint_msg.header.stamp = this->now();
    waypoint_msg.pose = current_waypoint;
    waypoint_publisher_->publish(waypoint_msg);

    publishWaypointMarker(current_waypoint);

    const double dx = tcp_xyz[0] - current_waypoint.position.x;
    const double dy = tcp_xyz[1] - current_waypoint.position.y;
    const double dz = tcp_xyz[2] - current_waypoint.position.z;
    const double waypoint_distance = std::sqrt(dx * dx + dy * dy + dz * dz);

    if (waypoint_distance <= waypoint_tolerance_m_ && waypoint_index_ < waypoints_.size() - 1) {
      waypoint_index_++;
      return;
    }

    if (waypoint_index_ >= waypoints_.size() - 1) {
      const auto &goal_waypoint = waypoints_.back();

      const double gx = tcp_xyz[0] - goal_waypoint.position.x;
      const double gy = tcp_xyz[1] - goal_waypoint.position.y;
      const double gz = tcp_xyz[2] - goal_waypoint.position.z;
      const double goal_distance = std::sqrt(gx * gx + gy * gy + gz * gz);

      if (goal_distance <= goal_tolerance_m_) {
        is_active_ = false;
        publishStatus("DONE_OK");
        publishStatus("IDLE");
      }
    }
  }

  void publishStartAndGoalMarkers(
    const std::array<double, 3> &start_xyz,
    const std::array<double, 3> &goal_xyz)
  {
    visualization_msgs::msg::Marker start_marker;
    start_marker.header.frame_id = base_frame_;
    start_marker.header.stamp = this->now();
    start_marker.ns = "hyrrt_start";
    start_marker.id = 0;
    start_marker.type = visualization_msgs::msg::Marker::SPHERE;
    start_marker.action = visualization_msgs::msg::Marker::ADD;
    start_marker.pose.orientation.w = 1.0;
    start_marker.pose.position.x = start_xyz[0];
    start_marker.pose.position.y = start_xyz[1];
    start_marker.pose.position.z = start_xyz[2];
    start_marker.scale.x = 0.03;
    start_marker.scale.y = 0.03;
    start_marker.scale.z = 0.03;
    start_marker.color.b = 1.0;
    start_marker.color.a = 1.0;
    start_marker_publisher_->publish(start_marker);
    last_start_marker_ = start_marker;

    visualization_msgs::msg::Marker goal_marker;
    goal_marker.header.frame_id = base_frame_;
    goal_marker.header.stamp = start_marker.header.stamp;
    goal_marker.ns = "hyrrt_goal";
    goal_marker.id = 0;
    goal_marker.type = visualization_msgs::msg::Marker::SPHERE;
    goal_marker.action = visualization_msgs::msg::Marker::ADD;
    goal_marker.pose.orientation.w = 1.0;
    goal_marker.pose.position.x = goal_xyz[0];
    goal_marker.pose.position.y = goal_xyz[1];
    goal_marker.pose.position.z = goal_xyz[2];
    goal_marker.scale.x = 0.03;
    goal_marker.scale.y = 0.03;
    goal_marker.scale.z = 0.03;
    goal_marker.color.g = 1.0;
    goal_marker.color.a = 1.0;
    goal_marker_publisher_->publish(goal_marker);
    last_goal_marker_ = goal_marker;
  }

  void publishPathMarker(const nav_msgs::msg::Path &path_msg)
  {
    visualization_msgs::msg::Marker path_marker;
    path_marker.header = path_msg.header;
    path_marker.ns = "hyrrt_path";
    path_marker.id = 0;
    path_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    path_marker.action = visualization_msgs::msg::Marker::ADD;
    path_marker.pose.orientation.w = 1.0;
    path_marker.scale.x = 0.01;
    path_marker.color.g = 0.8f;
    path_marker.color.b = 1.0f;
    path_marker.color.a = 1.0f;

    for (const auto &pose_stamped : path_msg.poses) {
      geometry_msgs::msg::Point point;
      point.x = pose_stamped.pose.position.x;
      point.y = pose_stamped.pose.position.y;
      point.z = pose_stamped.pose.position.z;
      path_marker.points.push_back(point);
    }

    path_marker_publisher_->publish(path_marker);
    last_path_marker_ = path_marker;
  }

  void publishTreeMarker()
  {
    ob::PlannerData planner_data(space_information_);
    planner_->getPlannerData(planner_data);

    visualization_msgs::msg::Marker tree_marker;
    tree_marker.header.frame_id = base_frame_;
    tree_marker.header.stamp = this->now();
    tree_marker.ns = "hyrrt_tree";
    tree_marker.id = 0;
    tree_marker.type = visualization_msgs::msg::Marker::LINE_LIST;
    tree_marker.action = visualization_msgs::msg::Marker::ADD;
    tree_marker.scale.x = 0.003;
    tree_marker.color.r = 1.0f;
    tree_marker.color.a = 1.0f;

    std::vector<unsigned int> edge_indices;
    for (unsigned int from_index = 0; from_index < planner_data.numVertices(); ++from_index) {
      const auto &from_vertex = planner_data.getVertex(from_index);
      const ob::State *from_state = from_vertex.getState();
      auto *from_values = from_state
                            ->as<ob::HybridStateSpace::StateType>()
                            ->as<ob::RealVectorStateSpace::StateType>(0)
                            ->values;

      geometry_msgs::msg::Point from_point;
      from_point.x = from_values[0];
      from_point.y = from_values[1];
      from_point.z = from_values[2];

      edge_indices.clear();
      planner_data.getEdges(from_index, edge_indices);

      for (unsigned int to_index : edge_indices) {
        const auto &to_vertex = planner_data.getVertex(to_index);
        const ob::State *to_state = to_vertex.getState();
        auto *to_values = to_state
                            ->as<ob::HybridStateSpace::StateType>()
                            ->as<ob::RealVectorStateSpace::StateType>(0)
                            ->values;

        geometry_msgs::msg::Point to_point;
        to_point.x = to_values[0];
        to_point.y = to_values[1];
        to_point.z = to_values[2];

        tree_marker.points.push_back(from_point);
        tree_marker.points.push_back(to_point);
      }
    }

    tree_marker_publisher_->publish(tree_marker);
    last_tree_marker_ = tree_marker;
  }

  void publishWaypointMarker(const geometry_msgs::msg::Pose &waypoint)
  {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = base_frame_;
    marker.header.stamp = this->now();
    marker.ns = "hyrrt_wp";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::SPHERE;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose = waypoint;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.03;
    marker.scale.y = 0.03;
    marker.scale.z = 0.03;
    marker.color.r = 1.0f;
    marker.color.g = 1.0f;
    marker.color.a = 1.0f;
    waypoint_marker_publisher_->publish(marker);
  }

  void publishTcpMarker(const std::array<double, 3> &tcp_xyz)
  {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = base_frame_;
    marker.header.stamp = this->now();
    marker.ns = "hyrrt_tcp";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::SPHERE;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.orientation.w = 1.0;
    marker.pose.position.x = tcp_xyz[0];
    marker.pose.position.y = tcp_xyz[1];
    marker.pose.position.z = tcp_xyz[2];
    marker.scale.x = 0.03;
    marker.scale.y = 0.03;
    marker.scale.z = 0.03;
    marker.color.r = 1.0f;
    marker.color.a = 1.0f;
    tcp_marker_publisher_->publish(marker);
  }

  void republishLatchedArtifacts()
  {
    if (!last_path_msg_.header.frame_id.empty()) {
      path_publisher_->publish(last_path_msg_);
    }
    if (!last_tree_marker_.header.frame_id.empty()) {
      tree_marker_publisher_->publish(last_tree_marker_);
    }
    if (!last_path_marker_.header.frame_id.empty()) {
      path_marker_publisher_->publish(last_path_marker_);
    }
    if (!last_start_marker_.header.frame_id.empty()) {
      start_marker_publisher_->publish(last_start_marker_);
    }
    if (!last_goal_marker_.header.frame_id.empty()) {
      goal_marker_publisher_->publish(last_goal_marker_);
    }
  }

  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_publisher_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr waypoint_publisher_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_publisher_;

  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr tree_marker_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr path_marker_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr start_marker_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr goal_marker_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr waypoint_marker_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr tcp_marker_publisher_;

  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr target_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_subscription_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr command_subscription_;

  rclcpp::TimerBase::SharedPtr waypoint_timer_;
  rclcpp::TimerBase::SharedPtr republish_timer_;

  std::string base_frame_;
  std::string target_topic_;
  std::string joint_state_topic_;
  std::string command_topic_;
  std::string status_topic_;
  std::string waypoint_topic_;
  std::string path_topic_;

  std::string tree_marker_topic_;
  std::string path_marker_topic_;
  std::string start_marker_topic_;
  std::string goal_marker_topic_;
  std::string waypoint_marker_topic_;
  std::string tcp_marker_topic_;

  double planning_time_sec_{5.0};
  double max_cartesian_velocity_{0.15};
  double flow_step_sec_{0.02};
  double minimum_tm_sec_{0.5};
  double waypoint_tolerance_m_{0.05};
  double goal_tolerance_m_{0.05};
  double waypoint_publish_period_sec_{0.02};

  std::string urdf_path_;
  std::string ee_link_;
  std::vector<std::string> joint_names_;

  bool clamp_to_workspace_{false};

  bool is_busy_{false};
  bool is_planned_{false};
  bool is_active_{false};
  bool kdl_ready_{false};
  bool has_forward_kinematics_{false};
  bool has_goal_{false};

  std::array<double, 3> tcp_xyz_{0.0, 0.0, 0.30};
  std::vector<double> current_joint_positions_;

  std::array<double, 3> goal_xyz_{0.30, 0.0, 0.30};
  std::mutex goal_mutex_;

  KDL::Chain kdl_chain_;
  std::unique_ptr<KDL::ChainFkSolverPos_recursive> fk_solver_;

  std::vector<geometry_msgs::msg::Pose> waypoints_;
  std::size_t waypoint_index_{0};

  ob::StateSpacePtr hybrid_state_space_;
  oc::SpaceInformationPtr space_information_;
  ob::ProblemDefinitionPtr problem_definition_;
  std::shared_ptr<oc::HyRRT> planner_;

  nav_msgs::msg::Path last_path_msg_;
  visualization_msgs::msg::Marker last_tree_marker_;
  visualization_msgs::msg::Marker last_path_marker_;
  visualization_msgs::msg::Marker last_start_marker_;
  visualization_msgs::msg::Marker last_goal_marker_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<Mode2TrajectoryNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}