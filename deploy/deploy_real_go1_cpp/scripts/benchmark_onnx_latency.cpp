#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <yaml-cpp/yaml.h>

namespace fs = std::filesystem;

struct Args {
    fs::path config = "deploy/deploy_real_go1_cpp/config/go1.yaml";
    fs::path policy;
    std::string slot = "initial";
    std::string provider = "both";
    int warmup = 200;
    int repeat = 2000;
    int threads = 1;
    int device = 0;
};

struct PolicySpec {
    std::string label;
    fs::path path;
    std::string display_path;
};

constexpr std::array<int, 6> kTerms = {3, 3, 3, 12, 12, 12};

std::string session_input_name(Ort::Session& session, std::size_t index, Ort::AllocatorWithDefaultOptions& allocator) {
#if ORT_API_VERSION >= 13
    auto name = session.GetInputNameAllocated(index, allocator);
    return name.get();
#else
    char* name = session.GetInputName(index, allocator);
    std::string out = name;
    allocator.Free(name);
    return out;
#endif
}

std::string session_output_name(Ort::Session& session, std::size_t index, Ort::AllocatorWithDefaultOptions& allocator) {
#if ORT_API_VERSION >= 13
    auto name = session.GetOutputNameAllocated(index, allocator);
    return name.get();
#else
    char* name = session.GetOutputName(index, allocator);
    std::string out = name;
    allocator.Free(name);
    return out;
#endif
}

std::string replace_root(std::string s, const fs::path& root) {
    const std::string key = "{LEGGED_GYM_ROOT_DIR}";
    const auto pos = s.find(key);
    if (pos != std::string::npos) s.replace(pos, key.size(), root.string());
    return s;
}

fs::path root_from_config(fs::path cfg) {
    cfg = fs::absolute(cfg).parent_path();
    for (int i = 0; i < 3; ++i) cfg = cfg.parent_path();
    return cfg;
}

std::string display_path(const fs::path& path, const fs::path& root) {
    std::error_code ec;
    const auto rel_root = fs::relative(path, root, ec);
    if (!ec && !rel_root.empty() && rel_root.native().rfind("..", 0) != 0) return rel_root.string();
    ec.clear();
    const auto rel_cwd = fs::relative(path, fs::current_path(), ec);
    if (!ec && !rel_cwd.empty() && rel_cwd.native().rfind("..", 0) != 0) return rel_cwd.string();
    return path.filename().string();
}

std::vector<PolicySpec> policies(const Args& args, const YAML::Node& cfg, const fs::path& root) {
    if (!args.policy.empty()) return {{args.policy.stem().string(), args.policy, args.policy.string()}};
    const auto ps = cfg["policies"];
    if (!ps || !ps.IsMap()) throw std::runtime_error("Config has no policies map. Use --policy.");
    if (args.slot == "all") {
        std::vector<PolicySpec> out;
        for (const auto& p : ps) {
            const auto path = fs::path(replace_root(p.second.as<std::string>(), root));
            out.push_back({p.first.as<std::string>(), path, display_path(path, root)});
        }
        return out;
    }
    const std::string slot = args.slot == "initial" ? cfg["initial_policy"].as<std::string>() : args.slot;
    if (!ps[slot]) throw std::runtime_error("Policy slot is not listed in config: " + slot);
    const auto path = fs::path(replace_root(ps[slot].as<std::string>(), root));
    return {{slot, path, display_path(path, root)}};
}

double pct(std::vector<double> v, double q) {
    std::sort(v.begin(), v.end());
    const double x = (v.size() - 1) * q;
    const auto i = static_cast<std::size_t>(x);
    return v[i] * (1.0 - (x - i)) + v[std::min(i + 1, v.size() - 1)] * (x - i);
}

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto val = [&](const char* name) {
            if (++i >= argc) throw std::runtime_error(std::string("Missing value for ") + name);
            return std::string(argv[i]);
        };
        if (s == "--config") a.config = val("--config");
        else if (s == "--policy") a.policy = val("--policy");
        else if (s == "--slot") a.slot = val("--slot");
        else if (s == "--provider") a.provider = val("--provider");
        else if (s == "--warmup") a.warmup = std::stoi(val("--warmup"));
        else if (s == "--repeat") a.repeat = std::stoi(val("--repeat"));
        else if (s == "--threads") a.threads = std::stoi(val("--threads"));
        else if (s == "--device") a.device = std::stoi(val("--device"));
        else if (s == "-h" || s == "--help") {
            std::cout << "Usage: " << argv[0] << " [--config path] [--policy path] [--slot initial|all|up|down|left|right]\n"
                      << "       [--provider cpu|cuda|both] [--warmup N] [--repeat N] [--threads N] [--device N]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + s);
        }
    }
    if (a.provider != "cpu" && a.provider != "cuda" && a.provider != "both") throw std::runtime_error("--provider must be cpu, cuda, or both");
    return a;
}

std::vector<float> make_input(std::vector<float>& hist, const std::vector<float>& obs, int nobs, int hlen, int input_dim) {
    if (hlen > 1) std::move(hist.begin() + nobs, hist.end(), hist.begin());
    std::copy(obs.begin(), obs.end(), hist.end() - nobs);
    std::vector<float> input;
    input.reserve(static_cast<std::size_t>(input_dim));
    int start = 0;
    for (int dim : kTerms) {
        for (int frame = 0; frame < hlen; ++frame) {
            const int offset = frame * nobs + start;
            input.insert(input.end(), hist.begin() + offset, hist.begin() + offset + dim);
        }
        start += dim;
    }
    return input;
}

void bench(const PolicySpec& policy, const std::string& provider, const Args& args, int nobs, int nact) {
    const fs::path& path = policy.path;
    if (!fs::exists(path)) throw std::runtime_error("ONNX policy not found: " + path.string());

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "go1_latency");
    Ort::SessionOptions opt;
    opt.SetIntraOpNumThreads(args.threads);
    opt.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    if (provider == "cuda") {
        OrtCUDAProviderOptions cuda{};
        cuda.device_id = args.device;
        opt.AppendExecutionProvider_CUDA(cuda);
    }

    const auto t0 = std::chrono::steady_clock::now();
    Ort::Session session(env, path.c_str(), opt);
    const double create_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    Ort::AllocatorWithDefaultOptions alloc;
    std::string in_name = session_input_name(session, 0, alloc);
    std::string out_name = session_output_name(session, 0, alloc);
    const char* ins[] = {in_name.c_str()};
    const char* outs[] = {out_name.c_str()};
    auto input_type_info = session.GetInputTypeInfo(0);
    auto input_info = input_type_info.GetTensorTypeAndShapeInfo();
    auto shape = input_info.GetShape();
    if (shape.size() != 2) throw std::runtime_error("Expected rank-2 ONNX input");
    if (shape[1] <= 0) shape[1] = nobs;
    if (shape[1] % nobs) throw std::runtime_error("ONNX input dim is not a multiple of num_obs");

    const int input_dim = static_cast<int>(shape[1]);
    const int hlen = input_dim / nobs;
    std::vector<float> hist(static_cast<std::size_t>(hlen * nobs), 0.0f);
    std::vector<float> obs(static_cast<std::size_t>(nobs), 0.0f);
    std::array<int64_t, 2> input_shape = {1, input_dim};
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::size_t out_count = 0;
    for (int i = 0; i < args.warmup; ++i) {
        auto input = make_input(hist, obs, nobs, hlen, input_dim);
        auto tensor = Ort::Value::CreateTensor<float>(mem, input.data(), input.size(), input_shape.data(), input_shape.size());
        auto out = session.Run(Ort::RunOptions{nullptr}, ins, &tensor, 1, outs, 1);
        out_count = out[0].GetTensorTypeAndShapeInfo().GetElementCount();
        if (out_count != static_cast<std::size_t>(nact)) throw std::runtime_error("ONNX action output is not 12-dimensional");
    }

    std::vector<double> ms;
    ms.reserve(static_cast<std::size_t>(args.repeat));
    for (int i = 0; i < args.repeat; ++i) {
        const auto t = std::chrono::steady_clock::now();
        auto input = make_input(hist, obs, nobs, hlen, input_dim);
        auto tensor = Ort::Value::CreateTensor<float>(mem, input.data(), input.size(), input_shape.data(), input_shape.size());
        auto out = session.Run(Ort::RunOptions{nullptr}, ins, &tensor, 1, outs, 1);
        ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count());
        out_count = out[0].GetTensorTypeAndShapeInfo().GetElementCount();
        if (out_count != static_cast<std::size_t>(nact)) throw std::runtime_error("ONNX action output is not 12-dimensional");
    }

    const double mean = std::accumulate(ms.begin(), ms.end(), 0.0) / ms.size();
    double var = 0.0;
    for (double x : ms) var += (x - mean) * (x - mean);
    var /= ms.size();

    std::cout << "\n[" << policy.label << "] " << provider << "\n"
              << "  policy: " << policy.display_path << "\n"
              << "  size_mib: " << fs::file_size(path) / 1024.0 / 1024.0 << "\n"
              << "  input_shape: [1, " << input_dim << "], history_length: " << hlen << ", output_count: " << out_count << "\n"
              << "  session_create_ms: " << create_ms << "\n"
              << "  run_ms: mean=" << mean
              << ", std=" << std::sqrt(var)
              << ", min=" << *std::min_element(ms.begin(), ms.end())
              << ", p50=" << pct(ms, 0.50)
              << ", p90=" << pct(ms, 0.90)
              << ", p95=" << pct(ms, 0.95)
              << ", p99=" << pct(ms, 0.99)
              << ", max=" << *std::max_element(ms.begin(), ms.end()) << "\n"
              << "  approx_hz: " << 1000.0 / mean << "\n";
}

int main(int argc, char** argv) {
    try {
        const Args args = parse(argc, argv);
        YAML::Node cfg;
        fs::path root = fs::current_path();
        int nobs = 45;
        int nact = 12;
        if (args.policy.empty() || fs::exists(args.config)) {
            cfg = YAML::LoadFile(args.config.string());
            root = root_from_config(args.config);
            if (cfg["num_obs"]) nobs = cfg["num_obs"].as<int>();
            if (cfg["num_actions"]) nact = cfg["num_actions"].as<int>();
        }
        std::cout << "config: " << args.config << "\n"
                  << "warmup: " << args.warmup << ", repeat: " << args.repeat << ", threads: " << args.threads << "\n";
        for (const auto& p : policies(args, cfg, root)) {
            if (args.provider == "cpu" || args.provider == "both") bench(p, "cpu", args, nobs, nact);
            if (args.provider == "cuda" || args.provider == "both") bench(p, "cuda", args, nobs, nact);
        }
    } catch (const std::exception& e) {
        std::cerr << "benchmark failed: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
