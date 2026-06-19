#include "onnx_policy.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <stdexcept>

namespace go1_deploy {
namespace {
constexpr std::array<int, 6> kObsTermDims = {3, 3, 3, 12, 12, 12};
}

OnnxPolicy::OnnxPolicy(const std::filesystem::path& path, int num_obs, int num_actions)
    : env_(ORT_LOGGING_LEVEL_WARNING, "go1_policy"),
      session_(nullptr),
      num_obs_(num_obs),
      num_actions_(num_actions),
      input_dim_(num_obs),
      history_length_(1) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("ONNX policy does not exist: " + path.string());
    }

    session_options_.SetIntraOpNumThreads(1);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    session_ = Ort::Session(env_, path.c_str(), session_options_);

    auto input_name = session_.GetInputNameAllocated(0, allocator_);
    input_name_ = input_name.get();
    auto output_name = session_.GetOutputNameAllocated(0, allocator_);
    output_name_ = output_name.get();
    input_names_ = {input_name_.c_str()};
    output_names_ = {output_name_.c_str()};

    auto input_info = session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
    auto input_shape = input_info.GetShape();
    if (input_shape.size() != 2) {
        throw std::runtime_error("Expected ONNX input rank 2.");
    }
    if (input_shape[1] > 0) {
        input_dim_ = static_cast<int>(input_shape[1]);
    }
    if (input_dim_ % num_obs_ != 0) {
        throw std::runtime_error("ONNX input dim is not a multiple of num_obs.");
    }
    history_length_ = input_dim_ / num_obs_;
    history_.assign(static_cast<std::size_t>(history_length_ * num_obs_), 0.0f);
}

void OnnxPolicy::reset() {
    std::fill(history_.begin(), history_.end(), 0.0f);
}

std::vector<float> OnnxPolicy::make_input(const std::vector<float>& obs) {
    if (static_cast<int>(obs.size()) != num_obs_) {
        throw std::runtime_error("Observation size does not match policy num_obs.");
    }

    if (history_length_ > 1) {
        std::move(history_.begin() + num_obs_, history_.end(), history_.begin());
    }
    std::copy(obs.begin(), obs.end(), history_.end() - num_obs_);

    std::vector<float> input;
    input.reserve(static_cast<std::size_t>(input_dim_));
    int term_start = 0;
    for (int term_dim : kObsTermDims) {
        for (int frame = 0; frame < history_length_; ++frame) {
            const int offset = frame * num_obs_ + term_start;
            input.insert(input.end(), history_.begin() + offset, history_.begin() + offset + term_dim);
        }
        term_start += term_dim;
    }
    return input;
}

std::vector<float> OnnxPolicy::run(const std::vector<float>& obs) {
    std::vector<float> input = make_input(obs);
    std::array<int64_t, 2> input_shape = {1, input_dim_};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input.data(), input.size(), input_shape.data(), input_shape.size());

    auto outputs = session_.Run(
        Ort::RunOptions{nullptr},
        input_names_.data(),
        &input_tensor,
        1,
        output_names_.data(),
        1);

    if (outputs.empty() || !outputs[0].IsTensor()) {
        throw std::runtime_error("ONNX policy did not return a tensor action output.");
    }
    auto output_info = outputs[0].GetTensorTypeAndShapeInfo();
    std::size_t output_count = output_info.GetElementCount();
    if (output_count != static_cast<std::size_t>(num_actions_)) {
        throw std::runtime_error("ONNX action output is not 12-dimensional.");
    }
    const float* output_data = outputs[0].GetTensorData<float>();
    return std::vector<float>(output_data, output_data + output_count);
}

} // namespace go1_deploy
