// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "onnxruntime_cxx_api.h"
#include <algorithm>
#include <deque>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <yaml-cpp/yaml.h>

namespace isaaclab
{

class Algorithms
{
public:
    virtual std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs) = 0;
    virtual void reset() {}

    std::vector<float> get_action()
    {
        std::lock_guard<std::mutex> lock(act_mtx_);
        return action;
    }
    
    std::vector<float> action;
protected:
    std::mutex act_mtx_;
};

class OrtRunner : public Algorithms
{
public:
    OrtRunner(std::string model_path)
    {
        // Init Model
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "onnx_model");
        session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);

        session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);

        for (size_t i = 0; i < session->GetInputCount(); ++i) {
            auto input_name = session->GetInputNameAllocated(i, allocator);
            input_names.push_back(input_name.release());
            try {
                Ort::TypeInfo input_type = session->GetInputTypeInfo(i);
                input_shapes.push_back(input_type.GetTensorTypeAndShapeInfo().GetShape());
            } catch (const std::exception&) {
                input_shapes.emplace_back();
            }
        }

        // Get first output name. The action tensor is the first output for exported policies.
        auto output_name = session->GetOutputNameAllocated(0, allocator);
        output_names.push_back(output_name.release());
    }

    std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs)
    {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        // make sure all input names are in obs
        for (const auto& name : input_names) {
            if (obs.find(name) == obs.end()) {
                throw std::runtime_error("Input name " + std::string(name) + " not found in observations.");
            }
        }

        // Create input tensors
        std::vector<Ort::Value> input_tensors;
        std::vector<std::vector<int64_t>> runtime_input_shapes;
        for(int i(0); i<input_names.size(); ++i)
        {
            const std::string name_str(input_names[i]);
            auto& input_data = obs.at(name_str);
            auto input_shape = input_shapes[i];
            bool use_fallback_shape = input_shape.empty();
            size_t expected_input_size = 1;
            for (const auto dim : input_shape) {
                if (dim <= 0) {
                    use_fallback_shape = true;
                    break;
                }
                expected_input_size *= static_cast<size_t>(dim);
            }
            if (!use_fallback_shape && expected_input_size != input_data.size()) {
                use_fallback_shape = true;
            }
            if (use_fallback_shape) {
                input_shape = {1, static_cast<int64_t>(input_data.size())};
            }
            runtime_input_shapes.push_back(input_shape);
            auto input_tensor = Ort::Value::CreateTensor<float>(
                memory_info,
                input_data.data(),
                input_data.size(),
                runtime_input_shapes.back().data(),
                runtime_input_shapes.back().size()
            );
            input_tensors.push_back(std::move(input_tensor));
        }

        // Run the model
        auto output_tensor = session->Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), input_tensors.size(), output_names.data(), 1);

        // Copy output data
        auto floatarr = output_tensor.front().GetTensorMutableData<float>();
        const auto output_size = output_tensor.front().GetTensorTypeAndShapeInfo().GetElementCount();
        if (output_size == 0) {
            throw std::runtime_error("ONNX policy produced an empty action tensor.");
        }
        std::lock_guard<std::mutex> lock(act_mtx_);
        action.resize(output_size);
        std::memcpy(action.data(), floatarr, output_size * sizeof(float));
        return action;
    }

private:
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<const char*> input_names;
    std::vector<const char*> output_names;

    std::vector<std::vector<int64_t>> input_shapes;
};

class ParkourStudentRunner : public Algorithms
{
public:
    ParkourStudentRunner(
        const std::filesystem::path& policy_path,
        const std::filesystem::path& depth_encoder_path,
        const YAML::Node& cfg = YAML::Node())
    : ort_env_(ORT_LOGGING_LEVEL_WARNING, "parkour_student")
    {
        session_options_.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);

        load_model(policy_model_, policy_path);
        load_model(depth_model_, depth_encoder_path);

        proprioception_group_ = yaml_value<std::string>(cfg, "proprioception_group", "proprioception");
        policy_obs_input_name_ = yaml_value<std::string>(cfg, "policy_obs_input_name", "obs");
        policy_latent_input_name_ = yaml_value<std::string>(cfg, "policy_latent_input_name", "scandots_latent");
        depth_image_input_name_ = yaml_value<std::string>(cfg, "depth_image_input_name", "depth_image");
        depth_proprioception_input_name_ = yaml_value<std::string>(cfg, "depth_proprioception_input_name", "proprioception");

        num_prop_ = yaml_value<std::size_t>(cfg, "num_prop", 53);
        num_scan_ = yaml_value<std::size_t>(cfg, "num_scan", 132);
        num_priv_explicit_ = yaml_value<std::size_t>(cfg, "num_priv_explicit", 9);
        num_priv_latent_ = yaml_value<std::size_t>(cfg, "num_priv_latent", 29);
        num_hist_ = yaml_value<std::size_t>(cfg, "num_hist", 10);
        depth_latent_dim_ = yaml_value<std::size_t>(cfg, "depth_latent_dim", 32);
        depth_yaw_dim_ = yaml_value<std::size_t>(cfg, "depth_yaw_dim", 2);
        yaw_scale_ = yaml_value<float>(cfg, "yaw_scale", 1.5f);
        depth_provider_ = yaml_value<std::string>(cfg, "depth_provider", "constant");
        depth_constant_ = yaml_value<float>(cfg, "depth_constant", 0.5f);
        depth_binary_path_ = yaml_value<std::string>(cfg, "depth_binary_path", "/tmp/parkour_depth_image_float32.bin");
        depth_update_interval_ = std::max<std::size_t>(1, yaml_value<std::size_t>(cfg, "depth_update_interval", 5));
        debug_log_interval_ = yaml_value<std::size_t>(cfg, "debug_log_interval", 50);

        obs_dim_ = num_prop_ + num_scan_ + num_priv_explicit_ + num_priv_latent_ + num_prop_ * num_hist_;
        history_offset_ = num_prop_ + num_scan_ + num_priv_explicit_ + num_priv_latent_;
        prop_history_.assign(num_hist_, std::vector<float>(num_prop_, 0.0f));

        depth_image_size_ = yaml_depth_size(cfg);
        if (depth_image_size_ == 0) {
            depth_image_size_ = input_size_or_default(depth_model_, depth_image_input_name_, 87 * 58);
        }
        constant_depth_image_.assign(depth_image_size_, depth_constant_);
    }

    void reset() override
    {
        prop_history_.assign(num_hist_, std::vector<float>(num_prop_, 0.0f));
        cached_depth_output_.clear();
        cached_depth_image_.clear();
        depth_counter_ = 0;
        debug_counter_ = 0;
    }

    std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs) override
    {
        const auto prop_it = obs.find(proprioception_group_);
        if (prop_it == obs.end()) {
            throw std::runtime_error("ParkourStudentRunner: observation group '" + proprioception_group_ + "' not found.");
        }
        const auto& proprioception = prop_it->second;
        if (proprioception.size() != num_prop_) {
            throw std::runtime_error(
                "ParkourStudentRunner: proprioception size " + std::to_string(proprioception.size()) +
                " does not match expected " + std::to_string(num_prop_) + ".");
        }

        auto policy_obs = build_policy_observation(proprioception);
        auto depth_image = current_depth_image();

        const auto expected_depth_output = depth_latent_dim_ + depth_yaw_dim_;
        if (cached_depth_output_.size() < expected_depth_output || depth_counter_ % depth_update_interval_ == 0) {
            std::unordered_map<std::string, std::vector<float>> depth_inputs;
            depth_inputs[depth_image_input_name_] = depth_image;
            depth_inputs[depth_proprioception_input_name_] = proprioception;
            cached_depth_output_ = run_model(depth_model_, depth_inputs);
            cached_depth_image_ = depth_image;
        } else if (!cached_depth_image_.empty()) {
            depth_image = cached_depth_image_;
        }
        ++depth_counter_;

        const auto& depth_output = cached_depth_output_;
        if (depth_output.size() < expected_depth_output) {
            throw std::runtime_error(
                "ParkourStudentRunner: depth encoder output size " + std::to_string(depth_output.size()) +
                " is smaller than expected " + std::to_string(expected_depth_output) + ".");
        }

        std::vector<float> depth_latent(depth_output.begin(), depth_output.begin() + depth_latent_dim_);
        if (depth_yaw_dim_ >= 2 && policy_obs.size() >= 8) {
            policy_obs[6] = yaw_scale_ * depth_output[depth_latent_dim_];
            policy_obs[7] = yaw_scale_ * depth_output[depth_latent_dim_ + 1];
        }

        std::unordered_map<std::string, std::vector<float>> policy_inputs;
        policy_inputs[policy_obs_input_name_] = std::move(policy_obs);
        policy_inputs[policy_latent_input_name_] = std::move(depth_latent);

        auto policy_action = run_model(policy_model_, policy_inputs);
        if (policy_action.empty()) {
            throw std::runtime_error("ParkourStudentRunner: policy produced an empty action tensor.");
        }
        maybe_log_debug(depth_image, depth_output, policy_action, proprioception);
        auto history_proprioception = proprioception;
        if (history_proprioception.size() >= 8) {
            history_proprioception[6] = 0.0f;
            history_proprioception[7] = 0.0f;
        }
        update_history(history_proprioception);

        std::lock_guard<std::mutex> lock(act_mtx_);
        action = std::move(policy_action);
        return action;
    }

private:
    struct OrtModel
    {
        std::unique_ptr<Ort::Session> session;
        std::vector<std::string> input_names;
        std::vector<std::string> output_names;
        std::vector<std::vector<int64_t>> input_shapes;
    };

    template <typename T>
    static T yaml_value(const YAML::Node& node, const std::string& key, T default_value)
    {
        if (node && node[key]) {
            return node[key].as<T>();
        }
        return default_value;
    }

    static std::size_t shape_element_count(const std::vector<int64_t>& shape)
    {
        if (shape.empty()) {
            return 0;
        }
        std::size_t count = 1;
        for (const auto dim : shape) {
            if (dim <= 0) {
                return 0;
            }
            count *= static_cast<std::size_t>(dim);
        }
        return count;
    }

    std::size_t yaml_depth_size(const YAML::Node& cfg) const
    {
        if (!cfg || !cfg["depth_shape"]) {
            return 0;
        }
        const auto depth_shape = cfg["depth_shape"].as<std::vector<int64_t>>();
        return shape_element_count(depth_shape);
    }

    void load_model(OrtModel& model, const std::filesystem::path& model_path)
    {
        if (!std::filesystem::exists(model_path)) {
            throw std::runtime_error("ONNX model not found: " + model_path.string());
        }

        model.session = std::make_unique<Ort::Session>(ort_env_, model_path.c_str(), session_options_);

        for (size_t i = 0; i < model.session->GetInputCount(); ++i) {
            auto input_name = model.session->GetInputNameAllocated(i, allocator_);
            model.input_names.emplace_back(input_name.get());
            try {
                Ort::TypeInfo input_type = model.session->GetInputTypeInfo(i);
                model.input_shapes.push_back(input_type.GetTensorTypeAndShapeInfo().GetShape());
            } catch (const std::exception&) {
                model.input_shapes.emplace_back();
            }
        }

        auto output_name = model.session->GetOutputNameAllocated(0, allocator_);
        model.output_names.emplace_back(output_name.get());
    }

    std::size_t input_size_or_default(const OrtModel& model, const std::string& input_name, std::size_t fallback) const
    {
        for (std::size_t i = 0; i < model.input_names.size(); ++i) {
            if (model.input_names[i] == input_name) {
                const auto size = shape_element_count(model.input_shapes[i]);
                return size > 0 ? size : fallback;
            }
        }
        return fallback;
    }

    std::vector<float> run_model(OrtModel& model, std::unordered_map<std::string, std::vector<float>>& inputs)
    {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        std::vector<const char*> input_name_ptrs;
        std::vector<Ort::Value> input_tensors;
        std::vector<std::vector<int64_t>> runtime_input_shapes;
        input_name_ptrs.reserve(model.input_names.size());
        input_tensors.reserve(model.input_names.size());
        runtime_input_shapes.reserve(model.input_names.size());

        for (std::size_t i = 0; i < model.input_names.size(); ++i) {
            const auto& name = model.input_names[i];
            auto input_it = inputs.find(name);
            if (input_it == inputs.end()) {
                throw std::runtime_error("ONNX input name '" + name + "' not found.");
            }

            auto input_shape = model.input_shapes[i];
            auto& input_data = input_it->second;
            const auto expected_input_size = shape_element_count(input_shape);
            if (expected_input_size == 0 || expected_input_size != input_data.size()) {
                input_shape = {1, static_cast<int64_t>(input_data.size())};
            }

            runtime_input_shapes.push_back(input_shape);
            input_name_ptrs.push_back(name.c_str());
            input_tensors.push_back(Ort::Value::CreateTensor<float>(
                memory_info,
                input_data.data(),
                input_data.size(),
                runtime_input_shapes.back().data(),
                runtime_input_shapes.back().size()
            ));
        }

        std::vector<const char*> output_name_ptrs;
        output_name_ptrs.reserve(model.output_names.size());
        for (const auto& name : model.output_names) {
            output_name_ptrs.push_back(name.c_str());
        }

        auto output_tensors = model.session->Run(
            Ort::RunOptions{nullptr},
            input_name_ptrs.data(),
            input_tensors.data(),
            input_tensors.size(),
            output_name_ptrs.data(),
            output_name_ptrs.size()
        );

        auto* output_data = output_tensors.front().GetTensorMutableData<float>();
        const auto output_size = output_tensors.front().GetTensorTypeAndShapeInfo().GetElementCount();
        std::vector<float> output(output_size, 0.0f);
        if (output_size > 0) {
            std::memcpy(output.data(), output_data, output_size * sizeof(float));
        }
        return output;
    }

    void update_history(const std::vector<float>& proprioception)
    {
        prop_history_.push_back(proprioception);
        while (prop_history_.size() > num_hist_) {
            prop_history_.pop_front();
        }
        while (prop_history_.size() < num_hist_) {
            prop_history_.push_front(proprioception);
        }
    }

    std::vector<float> build_policy_observation(const std::vector<float>& proprioception) const
    {
        std::vector<float> policy_obs(obs_dim_, 0.0f);
        std::copy(proprioception.begin(), proprioception.end(), policy_obs.begin());

        auto write_it = policy_obs.begin() + static_cast<std::ptrdiff_t>(history_offset_);
        for (const auto& history_frame : prop_history_) {
            write_it = std::copy(history_frame.begin(), history_frame.end(), write_it);
        }
        return policy_obs;
    }

    std::vector<float> current_depth_image()
    {
        if (depth_provider_ == "binary_file") {
            std::ifstream file(depth_binary_path_, std::ios::binary);
            if (file.is_open()) {
                std::vector<float> depth(depth_image_size_, depth_constant_);
                file.read(reinterpret_cast<char*>(depth.data()), static_cast<std::streamsize>(depth.size() * sizeof(float)));
                if (file.gcount() == static_cast<std::streamsize>(depth.size() * sizeof(float))) {
                    return depth;
                }
                warn_depth_file_once("Depth binary file has the wrong size; falling back to constant depth.");
            } else {
                warn_depth_file_once("Depth binary file is not available; falling back to constant depth.");
            }
        }
        return constant_depth_image_;
    }

    void warn_depth_file_once(const std::string& message)
    {
        if (!warned_depth_file_) {
            std::cerr << "ParkourStudentRunner: " << message << "\n";
            warned_depth_file_ = true;
        }
    }

    void maybe_log_debug(
        const std::vector<float>& depth_image,
        const std::vector<float>& depth_output,
        const std::vector<float>& policy_action,
        const std::vector<float>& proprioception)
    {
        if (debug_log_interval_ == 0 || (++debug_counter_ % debug_log_interval_) != 0) {
            return;
        }

        const auto depth_minmax = std::minmax_element(depth_image.begin(), depth_image.end());
        const auto action_minmax = std::minmax_element(policy_action.begin(), policy_action.end());
        const float yaw0 = depth_output.size() > depth_latent_dim_ ? depth_output[depth_latent_dim_] : 0.0f;
        const float yaw1 = depth_output.size() > depth_latent_dim_ + 1 ? depth_output[depth_latent_dim_ + 1] : 0.0f;

        std::cout << "Parkour debug: depth_norm=["
                  << *depth_minmax.first << ", " << *depth_minmax.second
                  << "] yaw=[" << yaw0 << ", " << yaw1
                  << "] cmd=[" << proprioception[9] << ", " << proprioception[10]
                  << "] contact=[" << proprioception[49] << ", " << proprioception[50]
                  << ", " << proprioception[51] << ", " << proprioception[52]
                  << "] policy_action=[" << *action_minmax.first << ", " << *action_minmax.second
                  << "]\n";
    }

    Ort::Env ort_env_;
    Ort::SessionOptions session_options_;
    Ort::AllocatorWithDefaultOptions allocator_;
    OrtModel policy_model_;
    OrtModel depth_model_;

    std::string proprioception_group_;
    std::string policy_obs_input_name_;
    std::string policy_latent_input_name_;
    std::string depth_image_input_name_;
    std::string depth_proprioception_input_name_;

    std::size_t num_prop_ = 53;
    std::size_t num_scan_ = 132;
    std::size_t num_priv_explicit_ = 9;
    std::size_t num_priv_latent_ = 29;
    std::size_t num_hist_ = 10;
    std::size_t depth_latent_dim_ = 32;
    std::size_t depth_yaw_dim_ = 2;
    std::size_t obs_dim_ = 753;
    std::size_t history_offset_ = 223;
    float yaw_scale_ = 1.5f;

    std::string depth_provider_ = "constant";
    float depth_constant_ = 0.5f;
    std::string depth_binary_path_;
    std::size_t depth_update_interval_ = 5;
    std::size_t depth_counter_ = 0;
    std::size_t depth_image_size_ = 87 * 58;
    std::vector<float> constant_depth_image_;
    std::vector<float> cached_depth_image_;
    std::vector<float> cached_depth_output_;
    bool warned_depth_file_ = false;
    std::size_t debug_log_interval_ = 50;
    std::size_t debug_counter_ = 0;

    std::deque<std::vector<float>> prop_history_;
};
};
