#include <openvino/openvino.hpp>
#include <openvino/pass/make_stateful.hpp>
#include <openvino/pass/manager.hpp>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_WIN32)
#define ANR_EXPORT extern "C" __declspec(dllexport)
#else
#define ANR_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {
constexpr size_t frame_size = 480;
constexpr size_t state_size = 45304;
void error_message(char* destination, size_t capacity, const char* stage, const char* message) noexcept {
    if (destination && capacity) {
        std::snprintf(destination, capacity, "OpenVINO %s: %s", stage, message);
        destination[capacity - 1] = '\0';
    }
}
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
struct Native {
    // Destruction is reverse declaration order: request/tensors before model/Core.
    ov::Core core;
    std::shared_ptr<ov::Model> model;
    ov::CompiledModel compiled;
    ov::Tensor audio;
    ov::Tensor attenuation;
    ov::Tensor output;
    ov::InferRequest request;
    bool failed = false;
};
bool supports(const std::vector<ov::PropertyName>& properties, const char* name) {
    return std::any_of(properties.begin(), properties.end(), [name](const auto& p) {
        return p == name && p.is_mutable();
    });
}
bool same_device(const std::string& actual, const std::string& selected) {
    if (actual == selected) return true;
    if (selected == "GPU.0" && actual == "GPU") return true;
    return selected == "NPU" && actual.rfind("NPU.", 0) == 0;
}
}

ANR_EXPORT int anr_ov_create(const unsigned char* bytes, size_t length, const char* selected,
                            void** handle, char* actual, size_t actual_capacity,
                            char* error, size_t error_capacity) noexcept {
    if (handle) *handle = nullptr;
    try {
        require(bytes && length && selected && handle && actual && actual_capacity, "invalid initialization arguments");
        const std::string device(selected);
        require(device == "CPU" || device == "NPU" ||
                    (device.rfind("GPU.", 0) == 0 && device.size() > 4 &&
                     std::all_of(device.begin() + 4, device.end(), [](unsigned char c) { return c >= '0' && c <= '9'; })),
                "select only CPU, GPU.<index>, or NPU; automatic/heterogeneous fallback is prohibited");
        auto native = std::make_unique<Native>();
        const auto available = native->core.get_available_devices();
        if (std::none_of(available.begin(), available.end(), [&](const auto& candidate) { return same_device(candidate, device); })) {
            std::string message = "requested explicit device " + device + " is unavailable; installed/usable devices:";
            for (const auto& candidate : available) message += " " + candidate;
            message += "; install the official OpenVINO device plugin and supported hardware driver. CPU fallback is disabled";
            throw std::runtime_error(message);
        }
        // Read the original validated ONNX directly. No ORT, GPU plan, IR cache,
        // external model conversion or duplicated network topology is involved.
        native->model = native->core.read_model(std::string(reinterpret_cast<const char*>(bytes), length), ov::Tensor{});
        auto& model = native->model;
        require(model->inputs().size() == 3 && model->outputs().size() == 3, "unexpected DFN3 input/output count");
        for (const auto& pair : std::vector<std::pair<const char*, size_t>>{
                 {"input_frame", frame_size}, {"states", state_size}, {"atten_lim_db", 1}}) {
            const auto port = model->input(pair.first);
            require(port.get_element_type() == ov::element::f32 && ov::shape_size(port.get_shape()) == pair.second,
                    "DFN3 input has unexpected type/shape");
        }
        require(model->output("enhanced_audio_frame").get_element_type() == ov::element::f32 &&
                    ov::shape_size(model->output("enhanced_audio_frame").get_shape()) == frame_size,
                "DFN3 enhanced audio output has unexpected type/shape");
        require(model->output("new_states").get_element_type() == ov::element::f32 &&
                    ov::shape_size(model->output("new_states").get_shape()) == state_size,
                "DFN3 state output has unexpected type/shape");
        auto lsnr = std::dynamic_pointer_cast<ov::op::v0::Result>(model->output("lsnr").get_node_shared_ptr());
        require(static_cast<bool>(lsnr), "DFN3 lsnr result not found");
        model->remove_result(lsnr);
        const auto state_input = std::dynamic_pointer_cast<ov::op::v0::Parameter>(model->input("states").get_node_shared_ptr());
        const auto state_output = std::dynamic_pointer_cast<ov::op::v0::Result>(model->output("new_states").get_node_shared_ptr());
        require(state_input && state_output, "DFN3 recurrent Parameter/Result pair not found");
        ov::pass::Manager passes;
        passes.register_pass<ov::pass::MakeStateful>(ov::pass::MakeStateful::ParamResPairs{{state_input, state_output}});
        passes.run_passes(model);
        model->validate_nodes_and_infer_types();
        require(model->inputs().size() == 2 && model->outputs().size() == 1 && model->get_variables().size() == 1,
                "MakeStateful did not internalize the complete DFN3 state");

        const auto supported = native->core.get_property(device, ov::supported_properties);
        ov::AnyMap config;
        if (supports(supported, ov::hint::performance_mode.name())) {
            config.emplace(ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
        }
        if (supports(supported, ov::hint::execution_mode.name())) {
            config.emplace(ov::hint::execution_mode(ov::hint::ExecutionMode::ACCURACY));
        }
        if (supports(supported, ov::hint::inference_precision.name())) {
            // NPU hardware uses native low precision: do not claim FP32 math.
            config.emplace(ov::hint::inference_precision(device == "NPU" ? ov::element::f16 : ov::element::f32));
        }
        native->compiled = native->core.compile_model(model, device, config);
        const auto executing = native->compiled.get_property(ov::execution_devices);
        require(executing.size() == 1 && same_device(executing.front(), device),
                "compiled model reported an unexpected execution device; hidden fallback is prohibited");
        require(executing.front().size() < actual_capacity, "execution device name exceeds caller buffer");
        std::memcpy(actual, executing.front().c_str(), executing.front().size() + 1);
        native->request = native->compiled.create_infer_request();
        native->audio = ov::Tensor(ov::element::f32, model->input("input_frame").get_shape());
        native->attenuation = ov::Tensor(ov::element::f32, model->input("atten_lim_db").get_shape());
        native->output = ov::Tensor(ov::element::f32, model->output("enhanced_audio_frame").get_shape());
        native->request.set_tensor("input_frame", native->audio);
        native->request.set_tensor("atten_lim_db", native->attenuation);
        native->request.set_tensor("enhanced_audio_frame", native->output);
        std::fill_n(native->audio.data<float>(), frame_size, 0.0f);
        native->attenuation.data<float>()[0] = 3.0f;
        // Reset via the State API, without mapping/downloading the state tensor.
        native->request.reset_state();
        *handle = native.release();
        return 0;
    } catch (const std::exception& e) {
        error_message(error, error_capacity, "initialization (check Runtime/ONNX frontend/device plugin installation)", e.what());
    } catch (...) {
        error_message(error, error_capacity, "initialization", "unknown native exception");
    }
    return -1;
}

ANR_EXPORT int anr_ov_frame(void* handle, const float* input, float* output,
                           float attenuation, int reset, char* error, size_t error_capacity) noexcept {
    auto* native = static_cast<Native*>(handle);
    try {
        require(native && input && output, "invalid frame arguments");
        require(!native->failed, "previous inference failed; recreate the reducer before reuse");
        if (reset) native->request.reset_state();
        std::memcpy(native->audio.data<float>(), input, frame_size * sizeof(float));
        native->attenuation.data<float>()[0] = attenuation;
        native->request.infer(); // Synchronous completion: no in-flight work on drop/error.
        std::memcpy(output, native->output.data<const float>(), frame_size * sizeof(float));
        return 0;
    } catch (const std::exception& e) {
        if (native) native->failed = true;
        error_message(error, error_capacity, "frame (recreate the reducer)", e.what());
    } catch (...) {
        if (native) native->failed = true;
        error_message(error, error_capacity, "frame (recreate the reducer)", "unknown native exception");
    }
    return -1;
}

ANR_EXPORT void anr_ov_destroy(void* handle) noexcept {
    delete static_cast<Native*>(handle);
}
