#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace go1_deploy {

class OnnxPolicy {
public:
    OnnxPolicy(const std::filesystem::path& path, int num_obs, int num_actions);

    std::vector<float> run(const std::vector<float>& obs);
    void reset();
    int input_dim() const { return input_dim_; }
    int history_length() const { return history_length_; }

private:
    std::vector<float> make_input(const std::vector<float>& obs);

    Ort::Env env_;
    Ort::SessionOptions session_options_;
    Ort::Session session_;
    Ort::AllocatorWithDefaultOptions allocator_;
    std::string input_name_;
    std::string output_name_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    int num_obs_;
    int num_actions_;
    int input_dim_;
    int history_length_;
    std::vector<float> history_;
};

} // namespace go1_deploy
