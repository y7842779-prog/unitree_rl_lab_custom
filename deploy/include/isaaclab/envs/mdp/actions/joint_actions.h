// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <eigen3/Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include "isaaclab/envs/manager_based_rl_env.h"
#include "isaaclab/manager/action_manager.h"
#include <algorithm>
#include <deque>

namespace isaaclab
{

class JointAction : public ActionTerm
{
public:
    JointAction(YAML::Node cfg, ManagerBasedRLEnv* env)
    :ActionTerm(cfg, env)
    {
        if(cfg["joint_ids"].IsNull()) {
            _action_dim = env->robot->data.joint_ids_map.size();
        } else {
            _joint_ids = cfg["joint_ids"].as<std::vector<int>>();
            _action_dim = _joint_ids.size();
        }
        _raw_actions.resize(_action_dim, 0.0f);
        _processed_actions.resize(_action_dim, 0.0f);
        if(!cfg["scale"].IsNull()) {
            _scale = cfg["scale"].as<std::vector<float>>();
        }
        if(!cfg["offset"].IsNull()) {
            _offset = cfg["offset"].as<std::vector<float>>();
        }
        if(!cfg["clip"].IsNull()) {
            _clip = cfg["clip"].as<std::vector<std::vector<float> >>();
        }
        if(cfg["use_delay"] && !cfg["use_delay"].IsNull()) {
            _use_delay = cfg["use_delay"].as<bool>();
        }
        if(cfg["action_delay_steps"] && !cfg["action_delay_steps"].IsNull()) {
            if(cfg["action_delay_steps"].IsSequence()) {
                const auto delay_steps = cfg["action_delay_steps"].as<std::vector<int>>();
                if(!delay_steps.empty()) {
                    _action_delay_steps = delay_steps.front();
                }
            } else {
                _action_delay_steps = cfg["action_delay_steps"].as<int>();
            }
        }
        if(cfg["history_length"] && !cfg["history_length"].IsNull()) {
            _history_length = cfg["history_length"].as<int>();
        }
        _history_length = std::max(_history_length, _action_delay_steps + 1);
        _action_history_buf.assign(_history_length, std::vector<float>(_action_dim, 0.0f));
    }

    virtual void process_actions(std::vector<float> actions)
    {
        // TODO: modify action by joint_ids
        if(_use_delay) {
            _action_history_buf.push_back(actions);
            while(static_cast<int>(_action_history_buf.size()) > _history_length) {
                _action_history_buf.pop_front();
            }
            int delayed_index = static_cast<int>(_action_history_buf.size()) - 1 - _action_delay_steps;
            delayed_index = std::clamp(delayed_index, 0, static_cast<int>(_action_history_buf.size()) - 1);
            _raw_actions = _action_history_buf[delayed_index];
        } else {
            _raw_actions = actions;
        }
        if(!_clip.empty())
        {
            for(int i(0); i<_action_dim; ++i) {
                _raw_actions[i] = std::clamp(_raw_actions[i], _clip[i][0], _clip[i][1]);
            }
        }
        for(int i(0); i<_action_dim; ++i)
        {
            if(!_scale.empty()) {
                _processed_actions[i] = _raw_actions[i] * _scale[i];
            } else {
                _processed_actions[i] = _raw_actions[i];
            }
            if(!_offset.empty()) {
                _processed_actions[i] += _offset[i];
            }
        }
    }


    int action_dim() 
    {
        return _action_dim;
    }

    std::vector<float> raw_actions() 
    {
        return _raw_actions;
    }
    
    std::vector<float> processed_actions() 
    {
        return _processed_actions;
    }

    void reset()
    {
        _raw_actions.assign(_action_dim, 0.0f);
        _action_history_buf.assign(_history_length, std::vector<float>(_action_dim, 0.0f));
    }

protected:
    int _action_dim;
    std::vector<int> _joint_ids;

    std::vector<float> _raw_actions;
    std::vector<float> _processed_actions;

    std::vector<float> _scale;
    std::vector<float> _offset;
    std::vector<std::vector<float> > _clip;
    bool _use_delay = false;
    int _action_delay_steps = 0;
    int _history_length = 1;
    std::deque<std::vector<float>> _action_history_buf;
};


class JointPositionAction : public JointAction
{
public:
    JointPositionAction(YAML::Node cfg, ManagerBasedRLEnv* env)
    :JointAction(cfg, env)
    {
    }
};

class JointVelocityAction : public JointAction
{
public:
    JointVelocityAction(YAML::Node cfg, ManagerBasedRLEnv* env)
    :JointAction(cfg, env)
    {
    }
};

REGISTER_ACTION(JointPositionAction);
REGISTER_ACTION(JointVelocityAction);

};
