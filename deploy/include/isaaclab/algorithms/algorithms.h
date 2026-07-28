// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "onnxruntime_cxx_api.h"
#include <cstring>
#include <iostream>
#include <mutex>
#include <stdexcept>

namespace isaaclab
{

class Algorithms
{
public:
    virtual std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs) = 0;

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
};
