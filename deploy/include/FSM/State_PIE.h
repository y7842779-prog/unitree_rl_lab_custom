// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"
#include "isaaclab/algorithms/pie_onnx_runner.h"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include <yaml-cpp/yaml.h>

class State_PIE : public FSMState
{
public:
    State_PIE(int state_mode, std::string state_string);

    void enter() override;
    void run() override;
    void exit() override;

private:
    std::vector<float> build_proprioception();
    std::vector<float> command_vector() const;
    void update_proprio_history(const std::vector<float>& proprio);
    std::vector<float> proprio_history_array() const;
    void update_depth_history();
    std::vector<float> depth_history_array() const;
    std::vector<float> load_depth_file() const;
    void maybe_log_debug(
        const std::vector<float>& command,
        const std::vector<float>& proprio,
        const std::vector<float>& history,
        const std::vector<float>& depth,
        const std::vector<float>& action) const;
    void policy_loop();
    bool bad_orientation(float limit_angle_rad) const;
    void publish_default_targets();
    void clamp_targets(std::vector<float>& targets) const;
    void ensure_policy_loaded();

    YAML::Node deploy_cfg_;
    YAML::Node pie_cfg_;
    std::filesystem::path policy_dir_;
    std::filesystem::path policy_path_;
    std::filesystem::path depth_binary_path_;

    std::unique_ptr<isaaclab::PIEOnnxRunner> policy_;

    std::vector<int> joint_ids_map_;
    std::vector<float> default_joint_pos_;
    std::vector<float> action_scale_;
    std::vector<float> stiffness_;
    std::vector<float> damping_;
    std::vector<std::vector<float>> joint_position_limits_;

    std::vector<std::vector<float>> proprio_frames_;
    std::vector<std::vector<float>> depth_frames_;
    std::vector<float> last_action_;
    std::vector<float> latest_targets_;

    mutable std::mutex target_mtx_;
    std::thread policy_thread_;
    std::atomic<bool> policy_thread_running_{false};
    std::atomic<bool> warned_policy_error_{false};
    mutable std::size_t last_depth_file_size_ = 0;
    mutable bool last_depth_read_complete_ = false;

    float step_dt_ = 0.02f;
    float keyboard_lin_vel_x_ = 0.4f;
    float keyboard_ang_vel_z_ = 0.4f;
    float joystick_lin_vel_x_ = 0.8f;
    float joystick_ang_vel_z_ = 1.2f;
    float zero_command_threshold_ = 0.05f;
    float depth_constant_ = 0.7f;
    std::size_t depth_update_period_steps_ = 5;
    std::size_t debug_log_interval_ = 50;
    std::size_t policy_step_ = 0;
    bool hold_stand_when_zero_ = true;
    bool enforce_joint_position_limits_ = true;
    std::string command_source_ = "keyboard";
    std::string depth_provider_ = "constant";
};

REGISTER_FSM(State_PIE)
