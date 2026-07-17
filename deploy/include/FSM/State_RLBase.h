// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"
#include <chrono>
#include <filesystem>
#include <fstream>

class State_RLBase : public FSMState
{
public:
    State_RLBase(int state_mode, std::string state_string);
    
    void enter()
    {
        // set gain
        for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i)
        {
            lowcmd->msg_.motor_cmd()[i].kp() = env->robot->data.joint_stiffness[i];
            lowcmd->msg_.motor_cmd()[i].kd() = env->robot->data.joint_damping[i];
            lowcmd->msg_.motor_cmd()[i].dq() = 0;
            lowcmd->msg_.motor_cmd()[i].tau() = 0;
        }

        env->robot->update();
        // Start policy thread
        policy_thread_running = true;
        policy_thread = std::thread([this]{
            using clock = std::chrono::high_resolution_clock;
            const std::chrono::duration<double> desiredDuration(env->step_dt);
            const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

            // Initialize timing
            auto sleepTill = clock::now() + dt;
            env->reset();

            while (policy_thread_running)
            {
                env->step();

                // Sleep
                std::this_thread::sleep_until(sleepTill);
                sleepTill += dt;
            }
        });

        start_lowstate_logger();
    }

    void run();
    
    void exit()
    {
        policy_thread_running = false;
        if (policy_thread.joinable()) {
            policy_thread.join();
        }
        stop_lowstate_logger();
    }

private:
    void configure_lowstate_logger(const YAML::Node& cfg, const std::string& state_string);
    void start_lowstate_logger();
    void write_lowstate_log_header();
    void write_lowstate_log_sample();
    void stop_lowstate_logger();

    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;
    std::vector<std::vector<float>> safe_joint_position_limits_;

    bool lowstate_log_enabled_ = false;
    double lowstate_log_rate_ = 200.0;
    std::string lowstate_log_path_template_;
    std::filesystem::path lowstate_log_path_;
    std::ofstream lowstate_log_file_;
    std::chrono::steady_clock::time_point lowstate_log_start_time_;
    std::chrono::steady_clock::time_point lowstate_log_next_sample_time_;
    std::chrono::steady_clock::duration lowstate_log_period_{std::chrono::milliseconds(5)};
    std::size_t lowstate_log_samples_ = 0;

    std::thread policy_thread;
    bool policy_thread_running = false;
};

REGISTER_FSM(State_RLBase)
