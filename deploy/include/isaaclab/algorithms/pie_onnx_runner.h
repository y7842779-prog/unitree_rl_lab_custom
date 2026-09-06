// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "onnxruntime_cxx_api.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace isaaclab
{

class PIEOnnxRunner
{
public:
    static constexpr std::size_t kProprioDim = 45;
    static constexpr std::size_t kProprioHistoryDim = 450;
    static constexpr std::size_t kDepthHistoryDim = 2 * 60 * 86;
    static constexpr std::size_t kMemoryDim = 128;
    static constexpr std::size_t kActionDim = 12;

    explicit PIEOnnxRunner(const std::filesystem::path& model_path)
    : ort_env_(ORT_LOGGING_LEVEL_WARNING, "pie_policy")
    {
        if (!std::filesystem::exists(model_path)) {
            throw std::runtime_error("PIE policy ONNX not found: " + model_path.string());
        }

        session_options_.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
        session_ = std::make_unique<Ort::Session>(ort_env_, model_path.c_str(), session_options_);
        validate_contract();
        reset();
    }

    void reset()
    {
        memory_h_.assign(kMemoryDim, 0.0f);
    }

    std::vector<float> act(
        std::vector<float>& proprio,
        std::vector<float>& proprio_history,
        std::vector<float>& depth_history)
    {
        require_size("proprio", proprio, kProprioDim);
        require_size("proprio_history", proprio_history, kProprioHistoryDim);
        require_size("depth_history", depth_history, kDepthHistoryDim);

        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        std::array<const char*, 4> input_names = {
            "proprio",
            "proprio_history",
            "depth_history",
            "memory_h_in",
        };
        std::array<const char*, 2> output_names = {
            "actions",
            "memory_h_out",
        };

        std::array<std::array<int64_t, 2>, 2> flat_shapes = {{
            {{1, static_cast<int64_t>(kProprioDim)}},
            {{1, static_cast<int64_t>(kProprioHistoryDim)}},
        }};
        std::array<int64_t, 4> depth_shape = {
            1,
            2,
            60,
            86,
        };
        std::array<int64_t, 3> memory_shape = {
            1,
            1,
            static_cast<int64_t>(kMemoryDim),
        };

        std::vector<Ort::Value> input_tensors;
        input_tensors.reserve(input_names.size());
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info, proprio.data(), proprio.size(), flat_shapes[0].data(), flat_shapes[0].size()));
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info, proprio_history.data(), proprio_history.size(), flat_shapes[1].data(), flat_shapes[1].size()));
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info, depth_history.data(), depth_history.size(), depth_shape.data(), depth_shape.size()));
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info, memory_h_.data(), memory_h_.size(), memory_shape.data(), memory_shape.size()));

        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names.data(),
            input_tensors.data(),
            input_tensors.size(),
            output_names.data(),
            output_names.size()
        );

        if (output_tensors.size() != output_names.size()) {
            throw std::runtime_error("PIE policy returned an unexpected number of outputs.");
        }

        auto actions = tensor_to_vector(output_tensors[0], "actions", kActionDim);
        auto memory_h_out = tensor_to_vector(output_tensors[1], "memory_h_out", kMemoryDim);
        memory_h_ = std::move(memory_h_out);
        return actions;
    }

private:
    static void require_size(const std::string& name, const std::vector<float>& values, std::size_t expected)
    {
        if (values.size() != expected) {
            throw std::runtime_error(
                "PIE input '" + name + "' has " + std::to_string(values.size()) +
                " values, expected " + std::to_string(expected) + ".");
        }
        if (!std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); })) {
            throw std::runtime_error("PIE input '" + name + "' contains NaN or infinity.");
        }
    }

    static std::vector<float> tensor_to_vector(Ort::Value& tensor, const std::string& name, std::size_t expected)
    {
        const auto count = tensor.GetTensorTypeAndShapeInfo().GetElementCount();
        if (count != expected) {
            throw std::runtime_error(
                "PIE output '" + name + "' has " + std::to_string(count) +
                " values, expected " + std::to_string(expected) + ".");
        }
        auto* data = tensor.GetTensorMutableData<float>();
        std::vector<float> values(count, 0.0f);
        if (count > 0) {
            std::memcpy(values.data(), data, count * sizeof(float));
        }
        if (!std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); })) {
            throw std::runtime_error("PIE output '" + name + "' contains NaN or infinity.");
        }
        return values;
    }

    static bool shape_matches(const std::vector<int64_t>& actual, const std::vector<int64_t>& expected)
    {
        if (actual.size() != expected.size()) {
            return false;
        }
        for (std::size_t i = 0; i < actual.size(); ++i) {
            if (actual[i] > 0 && actual[i] != expected[i]) {
                return false;
            }
        }
        return true;
    }

    void validate_contract()
    {
        const std::unordered_map<std::string, std::vector<int64_t>> expected_inputs = {
            {"proprio", {1, static_cast<int64_t>(kProprioDim)}},
            {"proprio_history", {1, static_cast<int64_t>(kProprioHistoryDim)}},
            {"depth_history", {1, 2, 60, 86}},
            {"memory_h_in", {1, 1, static_cast<int64_t>(kMemoryDim)}},
        };
        const std::unordered_map<std::string, std::vector<int64_t>> expected_outputs = {
            {"actions", {1, static_cast<int64_t>(kActionDim)}},
            {"memory_h_out", {1, 1, static_cast<int64_t>(kMemoryDim)}},
        };

        std::unordered_map<std::string, std::vector<int64_t>> actual_inputs;
        for (std::size_t i = 0; i < session_->GetInputCount(); ++i) {
            auto name = session_->GetInputNameAllocated(i, allocator_);
            auto shape = session_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            actual_inputs.emplace(name.get(), std::move(shape));
        }

        std::unordered_map<std::string, std::vector<int64_t>> actual_outputs;
        for (std::size_t i = 0; i < session_->GetOutputCount(); ++i) {
            auto name = session_->GetOutputNameAllocated(i, allocator_);
            auto shape = session_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            actual_outputs.emplace(name.get(), std::move(shape));
        }

        for (const auto& [name, expected_shape] : expected_inputs) {
            const auto it = actual_inputs.find(name);
            if (it == actual_inputs.end() || !shape_matches(it->second, expected_shape)) {
                throw std::runtime_error("PIE ONNX input contract mismatch for '" + name + "'.");
            }
        }
        for (const auto& [name, expected_shape] : expected_outputs) {
            const auto it = actual_outputs.find(name);
            if (it == actual_outputs.end() || !shape_matches(it->second, expected_shape)) {
                throw std::runtime_error("PIE ONNX output contract mismatch for '" + name + "'.");
            }
        }
    }

    Ort::Env ort_env_;
    Ort::SessionOptions session_options_;
    Ort::AllocatorWithDefaultOptions allocator_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<float> memory_h_;
};

} // namespace isaaclab
