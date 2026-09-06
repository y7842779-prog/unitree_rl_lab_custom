#include "FSM/State_PIE.h"

#include <eigen3/Eigen/Dense>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <spdlog/spdlog.h>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace
{
constexpr std::size_t kNumJoints = 12;
constexpr std::size_t kProprioHistoryLength = 10;
constexpr std::size_t kDepthHistoryLength = 2;
constexpr std::size_t kDepthFrameDim = 60 * 86;

const std::array<int, 6> kTermDims = {3, 3, 3, 12, 12, 12};

template <typename T>
T yaml_value(const YAML::Node& node, const std::string& key, T default_value)
{
    if (node && node[key]) {
        return node[key].as<T>();
    }
    return default_value;
}

template <typename T>
std::vector<T> yaml_vector(const YAML::Node& node, const std::string& key, std::vector<T> default_value)
{
    if (node && node[key]) {
        return node[key].as<std::vector<T>>();
    }
    return default_value;
}

bool command_is_zero(const std::vector<float>& command, float threshold)
{
    if (command.size() < 3) {
        return true;
    }
    return std::abs(command[0]) < threshold
        && std::abs(command[1]) < threshold
        && std::abs(command[2]) < threshold;
}

void require_size(const std::string& name, const std::vector<float>& values, std::size_t expected)
{
    if (values.size() != expected) {
        throw std::runtime_error(
            "PIE config '" + name + "' has " + std::to_string(values.size()) +
            " values, expected " + std::to_string(expected) + ".");
    }
}

std::pair<float, float> min_max(const std::vector<float>& values)
{
    if (values.empty()) {
        return {0.0f, 0.0f};
    }
    const auto [min_it, max_it] = std::minmax_element(values.begin(), values.end());
    return {*min_it, *max_it};
}

std::string format_vector(const std::vector<float>& values, int precision = 3)
{
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(precision) << "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            stream << ", ";
        }
        stream << values[i];
    }
    stream << "]";
    return stream.str();
}

} // namespace

State_PIE::State_PIE(int state_mode, std::string state_string)
: FSMState(state_mode, state_string)
{
    const auto cfg = param::config["FSM"][state_string];
    policy_dir_ = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());
    policy_path_ = policy_dir_ / "exported" / "policy.onnx";
    deploy_cfg_ = YAML::LoadFile(policy_dir_ / "params" / "deploy.yaml");
    pie_cfg_ = deploy_cfg_["pie"];

    joint_ids_map_ = yaml_vector<int>(deploy_cfg_, "joint_ids_map", {3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8});
    default_joint_pos_ = yaml_vector<float>(
        deploy_cfg_,
        "default_joint_pos",
        {-0.1f, 0.9f, -1.8f, 0.1f, 0.9f, -1.8f, -0.1f, 0.9f, -1.8f, 0.1f, 0.9f, -1.8f});
    action_scale_ = yaml_vector<float>(deploy_cfg_, "action_scale", std::vector<float>(kNumJoints, 0.25f));
    stiffness_ = yaml_vector<float>(deploy_cfg_, "stiffness", {20.0f, 20.0f, 40.0f, 20.0f, 20.0f, 40.0f, 20.0f, 20.0f, 40.0f, 20.0f, 20.0f, 40.0f});
    damping_ = yaml_vector<float>(deploy_cfg_, "damping", {1.0f, 1.0f, 2.0f, 1.0f, 1.0f, 2.0f, 1.0f, 1.0f, 2.0f, 1.0f, 1.0f, 2.0f});

    require_size("default_joint_pos", default_joint_pos_, kNumJoints);
    require_size("action_scale", action_scale_, kNumJoints);
    require_size("stiffness", stiffness_, kNumJoints);
    require_size("damping", damping_, kNumJoints);
    if (joint_ids_map_.size() != kNumJoints) {
        throw std::runtime_error("PIE config 'joint_ids_map' must contain 12 SDK motor indices.");
    }

    step_dt_ = yaml_value<float>(deploy_cfg_, "step_dt", 0.02f);
    command_source_ = yaml_value<std::string>(pie_cfg_, "command_source", "keyboard");
    keyboard_lin_vel_x_ = yaml_value<float>(pie_cfg_, "keyboard_lin_vel_x", 0.4f);
    keyboard_ang_vel_z_ = yaml_value<float>(pie_cfg_, "keyboard_ang_vel_z", 0.4f);
    joystick_lin_vel_x_ = yaml_value<float>(pie_cfg_, "joystick_lin_vel_x", 0.8f);
    joystick_ang_vel_z_ = yaml_value<float>(pie_cfg_, "joystick_ang_vel_z", 1.2f);
    zero_command_threshold_ = yaml_value<float>(pie_cfg_, "zero_command_threshold", 0.05f);
    hold_stand_when_zero_ = yaml_value<bool>(pie_cfg_, "hold_stand_when_zero", true);

    depth_provider_ = yaml_value<std::string>(pie_cfg_, "depth_provider", "constant");
    depth_constant_ = yaml_value<float>(pie_cfg_, "depth_constant", 0.7f);
    depth_update_period_steps_ = std::max<std::size_t>(
        1,
        yaml_value<std::size_t>(pie_cfg_, "depth_update_period_steps", 5));
    depth_binary_path_ = yaml_value<std::string>(
        pie_cfg_,
        "depth_binary_path",
        "/tmp/pie_depth_60x86_float32.bin");
    debug_log_interval_ = yaml_value<std::size_t>(pie_cfg_, "debug_log_interval", 50);

    enforce_joint_position_limits_ = yaml_value<bool>(deploy_cfg_["safety"], "enforce_joint_position_limits", true);
    if (deploy_cfg_["safety"] && deploy_cfg_["safety"]["joint_position_limits"]) {
        joint_position_limits_ = deploy_cfg_["safety"]["joint_position_limits"].as<std::vector<std::vector<float>>>();
        if (joint_position_limits_.size() != kNumJoints) {
            throw std::runtime_error("PIE safety.joint_position_limits must contain 12 ranges.");
        }
    }

    last_action_.assign(kNumJoints, 0.0f);
    latest_targets_ = default_joint_pos_;

    this->registered_checks.emplace_back(
        std::make_pair(
            [this]() -> bool { return this->bad_orientation(1.0f); },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_PIE::enter()
{
    publish_default_targets();
    last_action_.assign(kNumJoints, 0.0f);
    proprio_frames_.clear();
    depth_frames_.clear();
    policy_step_ = 0;
    warned_policy_error_.store(false, std::memory_order_relaxed);

    for (std::size_t i = 0; i < kNumJoints; ++i) {
        const int sdk_id = joint_ids_map_[i];
        auto& motor = lowcmd->msg_.motor_cmd()[sdk_id];
        motor.kp() = stiffness_[i];
        motor.kd() = damping_[i];
        motor.dq() = 0.0f;
        motor.tau() = 0.0f;
    }

    try {
        ensure_policy_loaded();
    } catch (const std::exception& exc) {
        spdlog::error("PIE policy is not ready: {}", exc.what());
        return;
    }

    policy_->reset();
    policy_thread_running_.store(true, std::memory_order_release);
    policy_thread_ = std::thread(&State_PIE::policy_loop, this);
}

void State_PIE::run()
{
    std::vector<float> targets;
    {
        std::lock_guard<std::mutex> lock(target_mtx_);
        targets = latest_targets_.empty() ? default_joint_pos_ : latest_targets_;
    }
    clamp_targets(targets);

    for (std::size_t i = 0; i < kNumJoints; ++i) {
        lowcmd->msg_.motor_cmd()[joint_ids_map_[i]].q() = targets[i];
    }
}

void State_PIE::exit()
{
    policy_thread_running_.store(false, std::memory_order_release);
    if (policy_thread_.joinable()) {
        policy_thread_.join();
    }
    if (policy_) {
        policy_->reset();
    }
}

void State_PIE::ensure_policy_loaded()
{
    if (!policy_) {
        policy_ = std::make_unique<isaaclab::PIEOnnxRunner>(policy_path_);
        spdlog::info("Loaded PIE policy: {}", policy_path_.string());
    }
}

void State_PIE::policy_loop()
{
    using clock = std::chrono::high_resolution_clock;
    const std::chrono::duration<double> desired_duration(step_dt_);
    const auto dt = std::chrono::duration_cast<clock::duration>(desired_duration);
    auto sleep_till = clock::now() + dt;

    while (policy_thread_running_.load(std::memory_order_acquire)) {
        try {
            if (policy_step_ % depth_update_period_steps_ == 0 || depth_frames_.empty()) {
                update_depth_history();
            }

            auto proprio = build_proprioception();
            update_proprio_history(proprio);

            std::vector<float> targets = default_joint_pos_;
            const std::vector<float> command = {proprio[6], proprio[7], proprio[8]};
            if (!hold_stand_when_zero_ || !command_is_zero(command, zero_command_threshold_)) {
                auto history = proprio_history_array();
                auto depth = depth_history_array();
                auto action = policy_->act(proprio, history, depth);
                maybe_log_debug(command, proprio, history, depth, action);
                for (std::size_t i = 0; i < kNumJoints; ++i) {
                    targets[i] = default_joint_pos_[i] + action_scale_[i] * action[i];
                }
                last_action_ = std::move(action);
            } else {
                last_action_.assign(kNumJoints, 0.0f);
            }

            clamp_targets(targets);
            {
                std::lock_guard<std::mutex> lock(target_mtx_);
                latest_targets_ = std::move(targets);
            }
            warned_policy_error_.store(false, std::memory_order_relaxed);
        } catch (const std::exception& exc) {
            publish_default_targets();
            if (!warned_policy_error_.exchange(true, std::memory_order_relaxed)) {
                spdlog::error("PIE policy loop error: {}", exc.what());
            }
        }

        ++policy_step_;
        std::this_thread::sleep_until(sleep_till);
        sleep_till += dt;
    }
}

std::vector<float> State_PIE::command_vector() const
{
    std::vector<float> command(3, 0.0f);
    const bool use_keyboard = command_source_ == "keyboard" || command_source_ == "auto";
    const bool use_joystick = command_source_ == "joystick" || command_source_ == "auto";

    if (use_keyboard && FSMState::keyboard) {
        const std::string key = FSMState::keyboard->key();
        if (key == "w" || key == "up") {
            command[0] = keyboard_lin_vel_x_;
        } else if (key == "s" || key == "down") {
            command[0] = -keyboard_lin_vel_x_;
        } else if (key == "a" || key == "left") {
            command[2] = keyboard_ang_vel_z_;
        } else if (key == "d" || key == "right") {
            command[2] = -keyboard_ang_vel_z_;
        }

        if (!command_is_zero(command, zero_command_threshold_) || command_source_ == "keyboard") {
            return command;
        }
    }

    if (use_joystick && FSMState::lowstate) {
        auto* joystick = &FSMState::lowstate->joystick;
        const float vx = std::abs(joystick->ly()) > 0.05f ? joystick->ly() * joystick_lin_vel_x_ : 0.0f;
        const float wz = std::abs(joystick->rx()) > 0.05f ? -joystick->rx() * joystick_ang_vel_z_ : 0.0f;
        command[0] = vx;
        command[2] = wz;
    }

    return command;
}

std::vector<float> State_PIE::build_proprioception()
{
    std::vector<float> obs;
    obs.reserve(isaaclab::PIEOnnxRunner::kProprioDim);
    const auto command = command_vector();

    std::lock_guard<std::mutex> lock(FSMState::lowstate->mutex_);
    const auto& msg = FSMState::lowstate->msg_;
    const auto& imu = msg.imu_state();
    const auto& motors = msg.motor_state();

    for (int i = 0; i < 3; ++i) {
        obs.push_back(imu.gyroscope()[i]);
    }

    Eigen::Quaternionf quat(
        imu.quaternion()[0],
        imu.quaternion()[1],
        imu.quaternion()[2],
        imu.quaternion()[3]);
    quat.normalize();
    const Eigen::Vector3f projected_gravity = quat.conjugate() * Eigen::Vector3f(0.0f, 0.0f, -1.0f);
    for (int i = 0; i < 3; ++i) {
        obs.push_back(projected_gravity[i]);
    }

    obs.insert(obs.end(), command.begin(), command.end());

    for (std::size_t i = 0; i < kNumJoints; ++i) {
        obs.push_back(motors[joint_ids_map_[i]].q() - default_joint_pos_[i]);
    }
    for (std::size_t i = 0; i < kNumJoints; ++i) {
        obs.push_back(motors[joint_ids_map_[i]].dq());
    }
    obs.insert(obs.end(), last_action_.begin(), last_action_.end());

    if (obs.size() != isaaclab::PIEOnnxRunner::kProprioDim) {
        throw std::runtime_error("Internal PIE proprioception size mismatch.");
    }
    return obs;
}

void State_PIE::update_proprio_history(const std::vector<float>& proprio)
{
    if (proprio_frames_.empty()) {
        proprio_frames_.assign(kProprioHistoryLength, proprio);
        return;
    }
    proprio_frames_.push_back(proprio);
    while (proprio_frames_.size() > kProprioHistoryLength) {
        proprio_frames_.erase(proprio_frames_.begin());
    }
}

std::vector<float> State_PIE::proprio_history_array() const
{
    if (proprio_frames_.size() != kProprioHistoryLength) {
        throw std::runtime_error("PIE proprioception history is not initialized.");
    }

    std::vector<float> history;
    history.reserve(isaaclab::PIEOnnxRunner::kProprioHistoryDim);
    std::size_t start = 0;
    for (const int dim : kTermDims) {
        for (const auto& frame : proprio_frames_) {
            history.insert(history.end(), frame.begin() + start, frame.begin() + start + dim);
        }
        start += static_cast<std::size_t>(dim);
    }
    return history;
}

void State_PIE::update_depth_history()
{
    std::vector<float> frame(kDepthFrameDim, depth_constant_);
    if (depth_provider_ == "binary_file") {
        frame = load_depth_file();
    }

    if (depth_frames_.empty()) {
        depth_frames_.assign(kDepthHistoryLength, frame);
        return;
    }
    depth_frames_.push_back(std::move(frame));
    while (depth_frames_.size() > kDepthHistoryLength) {
        depth_frames_.erase(depth_frames_.begin());
    }
}

std::vector<float> State_PIE::load_depth_file() const
{
    std::vector<float> frame(kDepthFrameDim, depth_constant_);
    std::ifstream file(depth_binary_path_, std::ios::binary);
    if (!file.is_open()) {
        return frame;
    }
    file.read(reinterpret_cast<char*>(frame.data()), static_cast<std::streamsize>(frame.size() * sizeof(float)));
    if (file.gcount() != static_cast<std::streamsize>(frame.size() * sizeof(float))) {
        return std::vector<float>(kDepthFrameDim, depth_constant_);
    }
    return frame;
}

std::vector<float> State_PIE::depth_history_array() const
{
    if (depth_frames_.size() != kDepthHistoryLength) {
        throw std::runtime_error("PIE depth history is not initialized.");
    }

    std::vector<float> history;
    history.reserve(isaaclab::PIEOnnxRunner::kDepthHistoryDim);
    for (const auto& frame : depth_frames_) {
        history.insert(history.end(), frame.begin(), frame.end());
    }
    return history;
}

void State_PIE::maybe_log_debug(
    const std::vector<float>& command,
    const std::vector<float>& proprio,
    const std::vector<float>& history,
    const std::vector<float>& depth,
    const std::vector<float>& action) const
{
    if (debug_log_interval_ == 0 || (policy_step_ % debug_log_interval_) != 0) {
        return;
    }

    const auto proprio_range = min_max(proprio);
    const auto depth_range = min_max(depth);
    const auto action_range = min_max(action);
    const std::size_t memory_size = policy_ ? policy_->memory_size() : 0;
    const float memory_norm = policy_ ? policy_->memory_norm() : 0.0f;

    std::ostringstream stream;
    stream << std::fixed << std::setprecision(3)
           << "[PIE]\n"
           << "command       = " << format_vector(command) << "\n\n"
           << "proprio size  = " << proprio.size() << "\n"
           << "history size  = " << history.size() << "\n"
           << "depth size    = " << depth.size() << "\n"
           << "memory size   = " << memory_size << "\n\n"
           << "proprio min/max = [" << proprio_range.first << ", " << proprio_range.second << "]\n"
           << "depth min/max   = [" << depth_range.first << ", " << depth_range.second << "]\n"
           << "memory norm     = " << memory_norm << "\n\n"
           << "action =\n"
           << format_vector(action) << "\n\n"
           << "action min/max = [" << action_range.first << ", " << action_range.second << "]";
    spdlog::info("{}", stream.str());
}

bool State_PIE::bad_orientation(float limit_angle_rad) const
{
    if (!FSMState::lowstate) {
        return false;
    }

    std::lock_guard<std::mutex> lock(FSMState::lowstate->mutex_);
    const auto& imu = FSMState::lowstate->msg_.imu_state();
    Eigen::Quaternionf quat(
        imu.quaternion()[0],
        imu.quaternion()[1],
        imu.quaternion()[2],
        imu.quaternion()[3]);
    quat.normalize();
    const Eigen::Vector3f projected_gravity = quat.conjugate() * Eigen::Vector3f(0.0f, 0.0f, -1.0f);
    const float cos_tilt = std::clamp(-projected_gravity.z(), -1.0f, 1.0f);
    return std::acos(cos_tilt) > limit_angle_rad;
}

void State_PIE::publish_default_targets()
{
    std::lock_guard<std::mutex> lock(target_mtx_);
    latest_targets_ = default_joint_pos_;
}

void State_PIE::clamp_targets(std::vector<float>& targets) const
{
    if (!enforce_joint_position_limits_ || joint_position_limits_.empty()) {
        return;
    }
    for (std::size_t i = 0; i < std::min(targets.size(), joint_position_limits_.size()); ++i) {
        if (joint_position_limits_[i].size() == 2) {
            targets[i] = std::clamp(targets[i], joint_position_limits_[i][0], joint_position_limits_[i][1]);
        }
    }
}
