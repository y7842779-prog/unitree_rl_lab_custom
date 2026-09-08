#include "FSM/State_ParkourStudent.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <limits>
#include <mutex>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>

namespace
{
std::atomic<float> g_parkour_forward_command{0.0f};
std::array<bool, 4> g_previous_foot_contacts = {false, false, false, false};

template <typename T>
T yaml_value(const YAML::Node& node, const std::string& key, T default_value)
{
    if (node && node[key]) {
        return node[key].as<T>();
    }
    return default_value;
}

std::vector<int> yaml_int_vector(const YAML::Node& node, const std::string& key, std::vector<int> default_value)
{
    if (node && node[key]) {
        return node[key].as<std::vector<int>>();
    }
    return default_value;
}

std::pair<float, float> roll_pitch_from_quat(const Eigen::Quaternionf& quat)
{
    const float qw = quat.w();
    const float qx = quat.x();
    const float qy = quat.y();
    const float qz = quat.z();

    const float sinr_cosp = 2.0f * (qw * qx + qy * qz);
    const float cosr_cosp = 1.0f - 2.0f * (qx * qx + qy * qy);
    const float roll = std::atan2(sinr_cosp, cosr_cosp);

    const float sinp = 2.0f * (qw * qy - qz * qx);
    const float pitch = std::asin(std::clamp(sinp, -1.0f, 1.0f));
    return {roll, pitch};
}

float forward_command(isaaclab::ManagerBasedRLEnv* env)
{
    const auto base_velocity_cfg = env->cfg["commands"]["base_velocity"];
    const auto ranges = base_velocity_cfg["ranges"];
    const float x_min = ranges["lin_vel_x"][0].as<float>(0.0f);
    const float x_max = ranges["lin_vel_x"][1].as<float>(0.8f);
    const float default_command = base_velocity_cfg["default_forward_velocity"].as<float>(0.0f);
    const float activation_command = base_velocity_cfg["activation_forward_velocity"].as<float>(0.3f);
    const float stop_command = base_velocity_cfg["stop_forward_velocity"].as<float>(0.0f);
    const float keyboard_step = base_velocity_cfg["keyboard_velocity_step"].as<float>(0.05f);
    const std::string command_source = base_velocity_cfg["command_source"].as<std::string>("auto");
    static bool initialized = false;
    static float command = 0.0f;
    static std::string last_key;

    if (!initialized || env->episode_length <= 1 || command < x_min || command > x_max) {
        command = std::clamp(default_command, x_min, x_max);
        initialized = true;
        last_key.clear();
    }

    const bool use_keyboard = command_source == "keyboard" || command_source == "auto";
    const bool use_joystick = command_source == "joystick" || command_source == "auto";

    if (use_keyboard && FSMState::keyboard) {
        const std::string key = FSMState::keyboard->key();
        if (!key.empty() && key != last_key) {
            bool changed = false;
            if (key == "w" || key == "up") {
                if (command < activation_command) {
                    command = activation_command;
                } else {
                    command = std::min(command + keyboard_step, x_max);
                }
                changed = true;
            } else if (key == "s" || key == "down") {
                if (command <= activation_command) {
                    command = stop_command;
                } else {
                    command = std::max(command - keyboard_step, activation_command);
                }
                changed = true;
            } else if (key == "x" || key == " ") {
                command = stop_command;
                changed = true;
            }
            if (changed) {
                spdlog::info("Parkour forward command: {:.2f} m/s", command);
            }
        }
        last_key = key;
    }

    auto* joystick = env->robot->data.joystick;
    if (use_joystick && joystick) {
        const float joy = joystick->ly();
        if (std::abs(joy) > 0.05f) {
            const float normalized = std::clamp(joy, 0.0f, 1.0f);
            command = x_min + normalized * (x_max - x_min);
        }
    }

    const float clamped_command = std::clamp(command, x_min, x_max);
    g_parkour_forward_command.store(clamped_command, std::memory_order_relaxed);
    return clamped_command;
}

std::vector<float> contact_fill(const YAML::Node& params)
{
    std::vector<float> fill(4, -0.5f);
    if (!FSMState::lowstate) {
        return fill;
    }

    const float threshold = yaml_value<float>(params, "contact_force_threshold", 20.0f);
    const auto contact_indices = yaml_int_vector(params, "contact_force_indices", {0, 1, 2, 3});

    std::lock_guard<std::mutex> lock(FSMState::lowstate->mutex_);
    const auto& force_est = FSMState::lowstate->msg_.foot_force_est();
    for (int i = 0; i < 4; ++i) {
        const int source_index = i < static_cast<int>(contact_indices.size()) ? contact_indices[i] : i;
        if (source_index >= 0 && source_index < 4) {
            const bool contact = static_cast<float>(force_est[source_index]) > threshold;
            const bool filtered_contact = contact || g_previous_foot_contacts[i];
            fill[i] = filtered_contact ? 0.5f : -0.5f;
            g_previous_foot_contacts[i] = contact;
        }
    }
    return fill;
}
}

namespace isaaclab
{

REGISTER_OBSERVATION(parkour_proprioception)
{
    std::vector<float> obs;
    obs.reserve(53);

    auto& data = env->robot->data;
    for (int i = 0; i < 3; ++i) {
        obs.push_back(data.root_ang_vel_b[i] * 0.25f);
    }

    const auto [roll, pitch] = roll_pitch_from_quat(data.root_quat_w.normalized());
    obs.push_back(roll);
    obs.push_back(pitch);

    obs.push_back(0.0f);
    obs.push_back(yaml_value<float>(params, "delta_yaw", 0.0f));
    obs.push_back(yaml_value<float>(params, "delta_next_yaw", 0.0f));
    obs.push_back(0.0f);
    const float command = forward_command(env);
    obs.push_back(yaml_value<bool>(params, "mirror_forward_command_to_demo_index", true) ? command : 0.0f);
    obs.push_back(command);
    obs.push_back(yaml_value<bool>(params, "terrain_is_parkour", true) ? 1.0f : 0.0f);
    obs.push_back(yaml_value<bool>(params, "terrain_is_flat", false) ? 1.0f : 0.0f);

    for (int i = 0; i < data.joint_pos.size(); ++i) {
        obs.push_back(data.joint_pos[i] - data.default_joint_pos[i]);
    }
    for (int i = 0; i < data.joint_vel.size(); ++i) {
        obs.push_back(data.joint_vel[i] * 0.05f);
    }

    auto last_action = env->action_manager->action();
    last_action.resize(12, 0.0f);
    obs.insert(obs.end(), last_action.begin(), last_action.begin() + 12);

    const auto contacts = contact_fill(params);
    obs.insert(obs.end(), contacts.begin(), contacts.end());

    if (obs.size() != 53) {
        obs.resize(53, 0.0f);
    }
    return obs;
}

}

State_ParkourStudent::State_ParkourStudent(int state_mode, std::string state_string)
: FSMState(state_mode, state_string)
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());
    auto deploy_cfg = YAML::LoadFile(policy_dir / "params" / "deploy.yaml");

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        deploy_cfg,
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::ParkourStudentRunner>(
        policy_dir / "exported" / "policy.onnx",
        policy_dir / "exported" / "depth_latest.onnx",
        deploy_cfg["parkour_student"]
    );

    auto safety_limits = env->cfg["safety"]["joint_position_limits"];
    enforce_joint_position_limits_ = yaml_value<bool>(env->cfg["safety"], "enforce_joint_position_limits", true);
    if (safety_limits && enforce_joint_position_limits_) {
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

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_ParkourStudent::enter()
{
    g_previous_foot_contacts = {false, false, false, false};

    for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i) {
        lowcmd->msg_.motor_cmd()[i].kp() = env->robot->data.joint_stiffness[i];
        lowcmd->msg_.motor_cmd()[i].kd() = env->robot->data.joint_damping[i];
        lowcmd->msg_.motor_cmd()[i].dq() = 0;
        lowcmd->msg_.motor_cmd()[i].tau() = 0;
    }

    env->robot->update();
    policy_thread_running = true;
    policy_thread = std::thread([this]{
        using clock = std::chrono::high_resolution_clock;
        const std::chrono::duration<double> desiredDuration(env->step_dt);
        const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

        auto sleepTill = clock::now() + dt;
        env->reset();

        while (policy_thread_running) {
            env->step();
            std::this_thread::sleep_until(sleepTill);
            sleepTill += dt;
        }
    });
}

void State_ParkourStudent::run()
{
    auto action = env->action_manager->processed_actions();
    const auto joint_count = env->robot->data.joint_ids_map.size();
    if (action.size() < joint_count) {
        static bool warned_action_size = false;
        if (!warned_action_size) {
            spdlog::error("Parkour policy action size {} is smaller than joint count {}.", action.size(), joint_count);
            warned_action_size = true;
        }
        return;
    }

    static bool warned_nonfinite_q_target = false;
    float q_target_min = std::numeric_limits<float>::max();
    float q_target_max = std::numeric_limits<float>::lowest();
    const bool hold_stand = std::abs(g_parkour_forward_command.load(std::memory_order_relaxed)) < 1.0e-4f;
    for (int i = 0; i < joint_count; ++i) {
        float q_target = hold_stand ? env->robot->data.default_joint_pos[i] : action[i];
        if (!std::isfinite(q_target)) {
            if (!warned_nonfinite_q_target) {
                spdlog::error("Parkour policy produced a non-finite q target. Falling back to default joint positions.");
                warned_nonfinite_q_target = true;
            }
            q_target = env->robot->data.default_joint_pos[i];
        }
        if (enforce_joint_position_limits_ && !safe_joint_position_limits_.empty()) {
            const auto& limit = safe_joint_position_limits_[i];
            q_target = std::clamp(q_target, limit[0], limit[1]);
        }
        q_target_min = std::min(q_target_min, q_target);
        q_target_max = std::max(q_target_max, q_target);
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = q_target;
    }

    static int debug_counter = 0;
    if (++debug_counter % 1000 == 0) {
        spdlog::info("Parkour q_target range: [{:.3f}, {:.3f}]", q_target_min, q_target_max);
    }
}

void State_ParkourStudent::exit()
{
    policy_thread_running = false;
    if (policy_thread.joinable()) {
        policy_thread.join();
    }
}
