// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"
#include "isaaclab/envs/manager_based_rl_env.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"
#include <chrono>
#include <filesystem>
#include <thread>
#include <vector>

class State_ParkourStudent : public FSMState
{
public:
    State_ParkourStudent(int state_mode, std::string state_string);

    void enter();
    void run();
    void exit();

private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;
    std::vector<std::vector<float>> safe_joint_position_limits_;
    bool enforce_joint_position_limits_ = true;

    std::thread policy_thread;
    bool policy_thread_running = false;
};

REGISTER_FSM(State_ParkourStudent)
