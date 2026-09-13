#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sstream>

namespace isaaclab
{

REGISTER_OBSERVATION(keyboard_velocity_commands)
{
    std::vector<float> obs(3, 0.0f);
    if(!FSMState::keyboard) return obs;

    const std::string key = FSMState::keyboard->key();
    const auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];

    const float x_min = cfg["lin_vel_x"][0].as<float>();
    const float x_max = cfg["lin_vel_x"][1].as<float>();
    const float y_min = cfg["lin_vel_y"][0].as<float>();
    const float y_max = cfg["lin_vel_y"][1].as<float>();
    const float yaw_min = cfg["ang_vel_z"][0].as<float>();
    const float yaw_max = cfg["ang_vel_z"][1].as<float>();

    if(key == "w" || key == "up") obs[0] = x_max;
    else if(key == "s" || key == "down") obs[0] = x_min;
    else if(key == "a" || key == "left") obs[1] = y_max;
    else if(key == "d" || key == "right") obs[1] = y_min;
    else if(key == "q") obs[2] = yaw_max;
    else if(key == "e") obs[2] = yaw_min;

    return obs;
}


REGISTER_OBSERVATION(history_base_ang_vel)
{
    return mdp::base_ang_vel(env, params);
}

REGISTER_OBSERVATION(history_projected_gravity)
{
    return mdp::projected_gravity(env, params);
}

REGISTER_OBSERVATION(history_velocity_commands)
{
    return mdp::velocity_commands(env, params);
}

REGISTER_OBSERVATION(history_keyboard_velocity_commands)
{
    return keyboard_velocity_commands(env, params);
}

REGISTER_OBSERVATION(history_joint_pos_rel)
{
    return mdp::joint_pos_rel(env, params);
}

REGISTER_OBSERVATION(history_joint_vel_rel)
{
    return mdp::joint_vel_rel(env, params);
}

REGISTER_OBSERVATION(history_last_action)
{
    return mdp::last_action(env, params);
}

}

namespace
{
std::string timestamp_string()
{
    const auto now = std::chrono::system_clock::now();
    const auto now_time = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    localtime_r(&now_time, &tm);

    std::ostringstream stream;
    stream << std::put_time(&tm, "%Y%m%d_%H%M%S");
    return stream.str();
}

std::string replace_all(std::string value, const std::string& from, const std::string& to)
{
    std::size_t pos = 0;
    while ((pos = value.find(from, pos)) != std::string::npos) {
        value.replace(pos, from.length(), to);
        pos += to.length();
    }
    return value;
}

template <typename ArrayType>
void write_array(std::ofstream& csv, const ArrayType& data)
{
    for (const auto& value : data) {
        csv << "," << value;
    }
}
}

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string) 
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    auto safety_limits = env->cfg["safety"]["joint_position_limits"];
    if (safety_limits) {
        safe_joint_position_limits_ = safety_limits.as<std::vector<std::vector<float>>>();
        const auto joint_count = env->robot->data.joint_ids_map.size();
        if (safe_joint_position_limits_.size() != joint_count) {
            throw std::runtime_error("safety.joint_position_limits size must match joint_ids_map size.");
        }
        for (const auto& limit : safe_joint_position_limits_) {
            if (limit.size() != 2 || limit[0] > limit[1]) {
                throw std::runtime_error("Each safety.joint_position_limits entry must be [lower, upper].");
            }
        }
    }
    configure_lowstate_logger(cfg, state_string);

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_RLBase::run()
{
    auto action = env->action_manager->processed_actions();
    const auto joint_count = env->robot->data.joint_ids_map.size();
    if (action.size() < joint_count) {
        static bool warned_action_size = false;
        if (!warned_action_size) {
            spdlog::error("Policy action size {} is smaller than joint count {}.", action.size(), joint_count);
            warned_action_size = true;
        }
        return;
    }

    static bool warned_nonfinite_q_target = false;
    for(int i(0); i < joint_count; i++) {
        float q_target = action[i];
        if (!std::isfinite(q_target)) {
            if (!warned_nonfinite_q_target) {
                spdlog::error("Policy produced a non-finite q target. Falling back to default joint positions.");
                warned_nonfinite_q_target = true;
            }
            q_target = env->robot->data.default_joint_pos[i];
        }
        if (!safe_joint_position_limits_.empty()) {
            const auto& limit = safe_joint_position_limits_[i];
            q_target = std::clamp(q_target, limit[0], limit[1]);
        }
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = q_target;
    }
    write_lowstate_log_sample();
}

void State_RLBase::configure_lowstate_logger(const YAML::Node& cfg, const std::string& state_string)
{
    auto log_cfg = cfg["lowstate_log"];
    if (!log_cfg) {
        return;
    }

    lowstate_log_enabled_ = !log_cfg["enable"] || log_cfg["enable"].as<bool>();
    if (!lowstate_log_enabled_) {
        return;
    }

    if (log_cfg["rate"]) {
        lowstate_log_rate_ = log_cfg["rate"].as<double>();
    }
    if (lowstate_log_rate_ <= 0.0) {
        throw std::runtime_error("FSM." + state_string + ".lowstate_log.rate must be positive.");
    }

    if (log_cfg["path"]) {
        lowstate_log_path_template_ = log_cfg["path"].as<std::string>();
    } else {
        lowstate_log_path_template_ = (param::proj_dir / "log" / (state_string + "_lowstate_{time}.csv")).string();
    }
}

void State_RLBase::start_lowstate_logger()
{
    if (!lowstate_log_enabled_) {
        return;
    }

    const auto path_string = replace_all(lowstate_log_path_template_, "{time}", timestamp_string());
    lowstate_log_path_ = std::filesystem::path(path_string);
    if (lowstate_log_path_.is_relative()) {
        lowstate_log_path_ = param::proj_dir / lowstate_log_path_;
    }
    if (lowstate_log_path_.has_parent_path()) {
        std::filesystem::create_directories(lowstate_log_path_.parent_path());
    }

    lowstate_log_file_.open(lowstate_log_path_);
    if (!lowstate_log_file_.is_open()) {
        throw std::runtime_error("Failed to open lowstate log file: " + lowstate_log_path_.string());
    }

    lowstate_log_file_ << std::setprecision(9);
    write_lowstate_log_header();
    lowstate_log_samples_ = 0;
    lowstate_log_start_time_ = std::chrono::steady_clock::now();
    lowstate_log_next_sample_time_ = lowstate_log_start_time_;
    lowstate_log_period_ = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(1.0 / lowstate_log_rate_)
    );

    spdlog::info("Lowstate policy log: {}", lowstate_log_path_.string());
}

void State_RLBase::write_lowstate_log_header()
{
    auto& csv = lowstate_log_file_;
    csv << "time,tick";

    for (int i = 0; i < 12; ++i) {
        csv << ",q" << i;
    }
    for (int i = 0; i < 12; ++i) {
        csv << ",dq" << i;
    }
    for (int i = 0; i < 12; ++i) {
        csv << ",tau" << i;
    }
    for (int i = 0; i < 12; ++i) {
        csv << ",temp" << i;
    }
    for (int i = 0; i < 12; ++i) {
        csv << ",lost" << i;
    }

    csv << ",imu_qw,imu_qx,imu_qy,imu_qz";
    csv << ",gyro_x,gyro_y,gyro_z";
    csv << ",acc_x,acc_y,acc_z";
    csv << ",roll,pitch,yaw";
    csv << ",foot_force0,foot_force1,foot_force2,foot_force3";
    csv << ",foot_force_est0,foot_force_est1,foot_force_est2,foot_force_est3";
    csv << ",joy_lx,joy_ly,joy_rx,joy_ry";
    csv << ",joy_cmd_vx_raw,joy_cmd_vy_raw,joy_cmd_yaw_raw";
    csv << "\n";
}

void State_RLBase::write_lowstate_log_sample()
{
    if (!lowstate_log_enabled_ || !lowstate_log_file_.is_open()) {
        return;
    }

    const auto now = std::chrono::steady_clock::now();
    if (now < lowstate_log_next_sample_time_) {
        return;
    }

    const double elapsed = std::chrono::duration<double>(now - lowstate_log_start_time_).count();
    while (lowstate_log_next_sample_time_ <= now) {
        lowstate_log_next_sample_time_ += lowstate_log_period_;
    }

    const float joy_lx = lowstate->joystick.lx();
    const float joy_ly = lowstate->joystick.ly();
    const float joy_rx = lowstate->joystick.rx();
    const float joy_ry = lowstate->joystick.ry();

    std::lock_guard<std::mutex> lock(lowstate->mutex_);
    const auto& msg = lowstate->msg_;
    const auto& imu = msg.imu_state();
    const auto& motors = msg.motor_state();
    auto& csv = lowstate_log_file_;

    csv << std::fixed << std::setprecision(6) << elapsed;
    csv << "," << msg.tick();

    for (int i = 0; i < 12; ++i) {
        csv << "," << motors[i].q();
    }
    for (int i = 0; i < 12; ++i) {
        csv << "," << motors[i].dq();
    }
    for (int i = 0; i < 12; ++i) {
        csv << "," << motors[i].tau_est();
    }
    for (int i = 0; i < 12; ++i) {
        csv << "," << static_cast<int>(motors[i].temperature());
    }
    for (int i = 0; i < 12; ++i) {
        csv << "," << motors[i].lost();
    }

    write_array(csv, imu.quaternion());
    write_array(csv, imu.gyroscope());
    write_array(csv, imu.accelerometer());
    write_array(csv, imu.rpy());
    write_array(csv, msg.foot_force());
    write_array(csv, msg.foot_force_est());

    csv << "," << joy_lx << "," << joy_ly << "," << joy_rx << "," << joy_ry;
    csv << "," << joy_ly << "," << -joy_lx << "," << -joy_rx;
    csv << "\n";

    ++lowstate_log_samples_;
    if (lowstate_log_samples_ % 200 == 0) {
        csv.flush();
    }
}

void State_RLBase::stop_lowstate_logger()
{
    if (!lowstate_log_file_.is_open()) {
        return;
    }

    lowstate_log_file_.flush();
    lowstate_log_file_.close();
    spdlog::info("Saved {} policy lowstate samples to {}", lowstate_log_samples_, lowstate_log_path_.string());
}
