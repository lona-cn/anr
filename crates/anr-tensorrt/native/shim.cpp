#include "shim.h"
#include <NvInfer.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nvrtc.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <initializer_list>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace nvinfer1;
constexpr size_t STATE = 45304;
constexpr size_t FRAME = 480;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
void driver(CUresult status, const char* operation) {
    if (status != CUDA_SUCCESS) {
        const char* text = nullptr;
        cuGetErrorString(status, &text);
        throw std::runtime_error(std::string(operation) + ": " + (text ? text : "CUDA driver error"));
    }
}
thread_local std::string last_error;
void report(char* out, size_t capacity, const char* text) noexcept {
    try { last_error = text; } catch (...) { last_error.clear(); }
    if (out && capacity) {
        std::strncpy(out, text, capacity - 1);
        out[capacity - 1] = 0;
    }
}
struct Logger final : ILogger {
    std::mutex mutex;
    std::string messages;
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) {
            try {
                std::lock_guard<std::mutex> lock(mutex);
                messages += message;
                messages += '\n';
            } catch (...) { /* Logging cannot propagate an allocation failure into TensorRT. */ }
        }
    }
    std::string snapshot() {
        std::lock_guard<std::mutex> lock(mutex);
        return messages;
    }
};
template<class T> T* checked(T* p, const char* message) { require(p != nullptr, message); return p; }
Dims dims(std::initializer_list<int64_t> sizes) {
    Dims d{};
    d.nbDims = static_cast<int>(sizes.size());
    int i = 0;
    for (auto v : sizes) d.d[i++] = v;
    return d;
}
int64_t volume(Dims d) {
    int64_t v = 1;
    for (int i = 0; i < d.nbDims; ++i) v *= d.d[i];
    return v;
}
struct Buffer {
    void* p = nullptr;
    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void allocate(size_t bytes) { check(cudaMalloc(&p, std::max(size_t(256), bytes)), "cudaMalloc"); }
    ~Buffer() { if (p) cudaFree(p); }
    float* floats() const { return static_cast<float*>(p); }
};
struct Binding { std::string name; uint32_t offset; int state; };
struct Engine {
    std::unique_ptr<ICudaEngine> engine;
    std::array<std::unique_ptr<IExecutionContext>, 2> context;
    std::array<Buffer, 2> workspace;
    std::vector<Binding> bindings;
    // Contexts reference workspace, engine and runtime. Destroy explicitly in
    // dependency order rather than relying on reverse member declaration order.
    ~Engine() { for (auto& c : context) c.reset(); }
};

std::vector<std::pair<uint32_t, uint32_t>> reads(const AnrDispatch& d) {
    const auto* p = d.p;
    switch (p[0]) {
    case 0: case 1: return {{p[2], p[6] * p[7] * p[8]}};
    case 2: return {{p[2], p[6] * p[7]}};
    case 3: return {{p[2], p[1]}, {p[4], p[1]}};
    case 4: case 5: return {{p[2], p[1]}};
    case 6: return {{p[2], p[1]}, {p[4], p[1]}};
    default: throw std::runtime_error("unknown Network opcode");
    }
}

// This is a lowering of the existing numeric Dispatch plan, not a second DFN3
// topology. Tensors are indexed by arena offsets; only edges crossing the NN
// range boundary become bindings. Intermediate tensors remain engine-local.
class Lowering {
    INetworkDefinition& net;
    const float* packed;
    size_t packed_count;
    std::deque<std::vector<float>> storage;
    std::map<uint32_t, ITensor*> tensors;
    std::vector<Binding>& bindings;

    Weights weights(uint32_t offset, size_t count) {
        require(offset <= packed_count && count <= packed_count - offset, "weight outside packed model");
        return {DataType::kFLOAT, packed + offset, static_cast<int64_t>(count)};
    }
    ITensor& reshape(ITensor& x, Dims shape) {
        require(volume(x.getDimensions()) == volume(shape), "dispatch tensor shape mismatch");
        auto* layer = checked(net.addShuffle(x), "TensorRT shuffle failed");
        layer->setReshapeDimensions(shape);
        return *layer->getOutput(0);
    }
    ITensor& transpose(ITensor& x, std::initializer_list<int> order) {
        auto* layer = checked(net.addShuffle(x), "TensorRT transpose failed");
        Permutation p{};
        int i = 0;
        for (int axis : order) p.order[i++] = axis;
        layer->setFirstTranspose(p);
        return *layer->getOutput(0);
    }
    ITensor& tensor(uint32_t offset, uint32_t len) {
        auto it = tensors.find(offset);
        if (it != tensors.end()) {
            require(volume(it->second->getDimensions()) == len, "scratch tensor length mismatch");
            return *it->second;
        }
        require(offset % 64 == 0, "TensorRT scratch input is not 256-byte aligned");
        std::string name = "scratch_in_" + std::to_string(offset);
        auto* t = checked(net.addInput(name.c_str(), DataType::kFLOAT, dims({len})), "TensorRT input failed");
        t->setAllowedFormats(1U << static_cast<unsigned>(TensorFormat::kLINEAR));
        tensors[offset] = t;
        bindings.push_back({name, offset, 0});
        return *t;
    }
    ITensor& activation(ITensor& x, uint32_t kind) {
        if (!kind) return x;
        auto type = kind == 1 ? ActivationType::kRELU : kind == 2 ? ActivationType::kSIGMOID : ActivationType::kTANH;
        return *checked(net.addActivation(x, type), "TensorRT activation failed")->getOutput(0);
    }
    ITensor& binary(ITensor& a, ITensor& b, ElementWiseOperation op) {
        return *checked(net.addElementWise(a, b, op), "TensorRT elementwise failed")->getOutput(0);
    }
    ITensor& constant(Dims shape, Weights data) {
        return *checked(net.addConstant(shape, data), "TensorRT constant failed")->getOutput(0);
    }
    ITensor& slice(ITensor& x, uint32_t start, uint32_t count) {
        return *checked(net.addSlice(x, dims({0, start}), dims({1, count}), dims({1, 1})), "TensorRT gate slice failed")->getOutput(0);
    }
    ITensor& matmul(ITensor& a, ITensor& b) {
        return *checked(net.addMatrixMultiply(a, MatrixOperation::kNONE, b, MatrixOperation::kNONE), "TensorRT matrix multiply failed")->getOutput(0);
    }
    ITensor& gru(const uint32_t* p) {
        auto& input = reshape(tensor(p[2], 256), dims({1, 256}));
        std::string name = "state_in_" + std::to_string(p[8]);
        auto* state = checked(net.addInput(name.c_str(), DataType::kFLOAT, dims({1, 256})), "TensorRT GRU state input failed");
        state->setAllowedFormats(1U << static_cast<unsigned>(TensorFormat::kLINEAR));
        bindings.push_back({name, p[8], 1});
        require((size_t(p[8]) * sizeof(float) + 32) % 256 == 0, "GRU state binding misaligned");
        // The shared plan stores [k][Wz,Wr,Wh,Rz,Rr,Rh][j]. Extract
        // [k][gate,j] for two explicit FP32 GEMMs; keep these builder weights
        // alive until buildSerializedNetwork has returned.
        std::array<ITensor*, 2> sums{};
        for (unsigned side = 0; side < 2; ++side) {
            storage.emplace_back(256 * 768);
            auto& w = storage.back();
            auto source = weights(p[4], 256 * 1536);
            const auto* src = static_cast<const float*>(source.values);
            for (unsigned k = 0; k < 256; ++k)
                std::copy_n(src + k * 1536 + side * 768, 768, w.data() + k * 768);
            auto& matrix = constant(dims({256, 768}), {DataType::kFLOAT, w.data(), static_cast<int64_t>(w.size())});
            auto& product = matmul(side ? *state : input, matrix);
            auto& bias = constant(dims({1, 768}), weights(p[10] + side * 768, 768));
            sums[side] = &binary(product, bias, ElementWiseOperation::kSUM);
        }
        auto& wz = slice(*sums[0], 0, 256);
        auto& wr = slice(*sums[0], 256, 256);
        auto& wh = slice(*sums[0], 512, 256);
        auto& rz = slice(*sums[1], 0, 256);
        auto& rr = slice(*sums[1], 256, 256);
        auto& rh = slice(*sums[1], 512, 256);
        auto& z = activation(binary(wz, rz, ElementWiseOperation::kSUM), 2);
        auto& r = activation(binary(wr, rr, ElementWiseOperation::kSUM), 2);
        auto& candidate = activation(binary(wh, binary(r, rh, ElementWiseOperation::kPROD), ElementWiseOperation::kSUM), 3);
        storage.push_back({1.0f});
        auto& one = constant(dims({1, 1}), {DataType::kFLOAT, storage.back().data(), 1});
        auto& result = binary(binary(binary(one, z, ElementWiseOperation::kSUB), candidate, ElementWiseOperation::kPROD),
            binary(z, *state, ElementWiseOperation::kPROD), ElementWiseOperation::kSUM);
        auto* copy = checked(net.addIdentity(result), "TensorRT state output identity failed")->getOutput(0);
        name = "state_out_" + std::to_string(p[8]);
        copy->setName(name.c_str());
        net.markOutput(*copy);
        copy->setAllowedFormats(1U << static_cast<unsigned>(TensorFormat::kLINEAR));
        bindings.push_back({name, p[8], 2});
        return result;
    }
public:
    Lowering(INetworkDefinition& n, const float* w, size_t count, std::vector<Binding>& b)
        : net(n), packed(w), packed_count(count), bindings(b) {}
    void lower(const AnrDispatch& d) {
        auto* p = d.p;
        ITensor* out = nullptr;
        switch (p[0]) {
        case 0: case 1: {
            auto& input = reshape(tensor(p[2], p[6] * p[7] * p[8]), dims({1, p[6], p[7], p[8]}));
            size_t count = size_t(p[9]) * (p[6] / p[18]) * p[12] * p[13];
            auto w = weights(p[4], count);
            Weights bias{DataType::kFLOAT, nullptr, 0};
            if (p[5] != UINT32_MAX) bias = weights(p[5], p[9]);
            if (!p[0]) {
                auto* conv = checked(net.addConvolutionNd(input, p[9], dims({p[12], p[13]}), w, bias), "TensorRT convolution failed");
                conv->setStrideNd(dims({p[14], p[15]}));
                conv->setPrePadding(dims({p[16], p[17]}));
                conv->setPostPadding(dims({p[16], p[17]}));
                conv->setNbGroups(p[18]);
                out = conv->getOutput(0);
            } else {
                auto* conv = checked(net.addDeconvolutionNd(input, p[9], dims({p[12], p[13]}), w, bias), "TensorRT deconvolution failed");
                conv->setStrideNd(dims({p[14], p[15]}));
                conv->setPrePadding(dims({p[16], p[17]}));
                conv->setPostPadding(dims({int64_t(p[16]) - (p[14] - 1), int64_t(p[17]) - (p[15] - 1)}));
                conv->setNbGroups(p[18]);
                out = conv->getOutput(0);
            }
            require(volume(out->getDimensions()) == p[1], "TensorRT convolution extent differs from Dispatch");
            out = &activation(*out, p[19]);
            break;
        }
        case 2: {
            auto& x = reshape(tensor(p[2], p[6] * p[7]), dims({p[6], 1, p[7]}));
            auto& w = constant(dims({p[6], p[7], p[8]}), weights(p[4], size_t(p[6]) * p[7] * p[8]));
            out = &activation(matmul(x, w), p[19]);
            break;
        }
        case 3: out = &binary(tensor(p[2], p[1]), tensor(p[4], p[1]), ElementWiseOperation::kSUM); break;
        case 4: {
            auto& x = reshape(tensor(p[2], p[1]), p[8] ? dims({p[7], p[6]}) : dims({p[6], p[7]}));
            out = &transpose(x, {1, 0});
            break;
        }
        case 5: out = &gru(p); break;
        case 6: {
            auto& a = reshape(tensor(p[2], p[1]), dims({96, 10}));
            auto& b = reshape(tensor(p[4], p[1]), dims({10, 96}));
            auto& sum = binary(transpose(a, {1, 0}), b, ElementWiseOperation::kSUM);
            auto& grouped = reshape(sum, dims({5, 2, 96}));
            out = &transpose(grouped, {0, 2, 1});
            break;
        }
        default: throw std::runtime_error("unknown Network opcode");
        }
        tensors[p[3]] = &reshape(*out, dims({p[1]}));
    }
    void output(uint32_t offset) {
        require(offset % 64 == 0, "TensorRT scratch output is not 256-byte aligned");
        auto* t = checked(net.addIdentity(*tensors.at(offset)), "TensorRT output identity failed")->getOutput(0);
        std::string name = "scratch_out_" + std::to_string(offset);
        t->setName(name.c_str());
        net.markOutput(*t);
        t->setAllowedFormats(1U << static_cast<unsigned>(TensorFormat::kLINEAR));
        bindings.push_back({name, offset, 0});
    }
};

struct Instance {
    Logger logger;
    std::unique_ptr<IRuntime> runtime;
    std::array<Engine, 2> engines;
    cudaStream_t stream = nullptr;
    CUmodule module = nullptr;
    std::array<CUfunction, 3> functions{};
    Buffer weights, scratch, parameters;
    std::array<Buffer, 2> state_allocation;
    std::array<float*, 2> state{};
    std::array<cudaGraph_t, 2> graph{};
    std::array<cudaGraphExec_t, 2> executable{};
    float* host_input = nullptr;
    float* host_output = nullptr;
    std::vector<AnrDispatch> dispatches;
    std::array<std::pair<size_t, size_t>, 2> ranges{};
    size_t attenuation_index = 0;
    uint32_t input_offset = 0, output_offset = 0, scratch_len = 0;
    unsigned parity = 0;
    float attenuation = 3.0f;
    bool failed = false;

    ~Instance() {
        // Drain on every path, including partial initialization, capture failure
        // and enqueue failure, before freeing any referenced object or pointer.
        if (stream) cudaStreamSynchronize(stream);
        for (auto e : executable) if (e) cudaGraphExecDestroy(e);
        for (auto g : graph) if (g) cudaGraphDestroy(g);
        for (auto& engine : engines) {
            for (auto& c : engine.context) c.reset();
            engine.engine.reset();
        }
        runtime.reset();
        if (module) cuModuleUnload(module);
        if (host_input) cudaFreeHost(host_input);
        if (host_output) cudaFreeHost(host_output);
        if (stream) cudaStreamDestroy(stream);
    }

    void compile_cuda(const char* source) {
        int device = 0;
        check(cudaGetDevice(&device), "cudaGetDevice");
        cudaDeviceProp properties{};
        check(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
        require(properties.major >= 7, "TensorRT requires a supported hardware CUDA GPU; no CPU fallback is available");
        nvrtcProgram program = nullptr;
        auto result = nvrtcCreateProgram(&program, source, "anr_dfn3.cu", 0, nullptr, nullptr);
        require(result == NVRTC_SUCCESS, std::string("nvrtcCreateProgram: ") + nvrtcGetErrorString(result));
        struct Guard { nvrtcProgram& p; ~Guard() { if (p) nvrtcDestroyProgram(&p); } } guard{program};
        std::string architecture = "--gpu-architecture=compute_" + std::to_string(properties.major) + std::to_string(properties.minor);
        const char* options[] = {architecture.c_str(), "--std=c++14", "--fmad=false"};
        result = nvrtcCompileProgram(program, 3, options);
        size_t length = 0;
        nvrtcGetProgramLogSize(program, &length);
        std::vector<char> log(std::max(size_t(1), length), 0);
        nvrtcGetProgramLog(program, log.data());
        require(result == NVRTC_SUCCESS, std::string("NVRTC CUDA DSP/FFT compilation: ") + nvrtcGetErrorString(result) + "\n" + log.data());
        require(nvrtcGetPTXSize(program, &length) == NVRTC_SUCCESS, "nvrtcGetPTXSize failed");
        std::vector<char> ptx(length);
        require(nvrtcGetPTX(program, ptx.data()) == NVRTC_SUCCESS, "nvrtcGetPTX failed");
        driver(cuModuleLoadData(&module, ptx.data()), "cuModuleLoadData");
        driver(cuModuleGetFunction(&functions[1], module, "dsp"), "CUDA DSP entry");
        driver(cuModuleGetFunction(&functions[2], module, "fft"), "CUDA FFT entry");
    }

    void build_engine(unsigned which, const float* packed, size_t count,
        const std::map<uint32_t, std::set<size_t>>& consumers) {
        auto& target = engines[which];
        std::unique_ptr<IBuilder> builder(checked(createInferBuilder(logger), "TensorRT builder creation failed"));
        std::unique_ptr<INetworkDefinition> network(checked(builder->createNetworkV2(1U << static_cast<unsigned>(NetworkDefinitionCreationFlag::kSTRONGLY_TYPED)), "TensorRT strongly typed network creation failed"));
        std::unique_ptr<IBuilderConfig> config(checked(builder->createBuilderConfig(), "TensorRT configuration failed"));
        config->clearFlag(BuilderFlag::kTF32);
        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, size_t(1) << 30);
        // A single stream makes ownership/capture deterministic; prohibit TRT
        // auxiliary execution streams rather than relying on implicit ordering.
        config->setMaxAuxStreams(0);
        config->setBuilderOptimizationLevel(3);
        config->setProfileStream(stream);
        Lowering lower(*network, packed, count, target.bindings);
        auto [begin, end] = ranges[which];
        for (size_t i = begin; i < end; ++i) lower.lower(dispatches[i]);
        for (size_t i = begin; i < end; ++i) {
            uint32_t offset = dispatches[i].p[3];
            auto found = consumers.find(offset);
            if (found == consumers.end()) continue;
            if (std::any_of(found->second.begin(), found->second.end(), [=](size_t consumer) { return consumer < begin || consumer >= end; })) lower.output(offset);
        }
        std::unique_ptr<IHostMemory> serialized(builder->buildSerializedNetwork(*network, *config));
        require(bool(serialized), "TensorRT engine build failed\n" + logger.snapshot());
        target.engine.reset(runtime->deserializeCudaEngine(serialized->data(), serialized->size()));
        require(bool(target.engine), "TensorRT engine deserialization failed\n" + logger.snapshot());
        for (const auto& binding : target.bindings) {
            require(target.engine->getTensorDataType(binding.name.c_str()) == DataType::kFLOAT &&
                target.engine->getTensorFormat(binding.name.c_str()) == TensorFormat::kLINEAR,
                "TensorRT engine I/O is not linear FP32: " + binding.name);
        }
        auto size = target.engine->getDeviceMemorySizeV2();
        require(size >= 0, "TensorRT reported invalid activation memory size");
        for (unsigned parity = 0; parity < 2; ++parity) {
            target.context[parity].reset(target.engine->createExecutionContext(ExecutionContextAllocationStrategy::kUSER_MANAGED));
            require(bool(target.context[parity]), "TensorRT execution context creation failed\n" + logger.snapshot());
            target.workspace[parity].allocate(static_cast<size_t>(size));
            target.context[parity]->setDeviceMemoryV2(target.workspace[parity].p, size);
            for (const auto& binding : target.bindings) {
                float* pointer = binding.state == 1 ? state[parity] + binding.offset
                    : binding.state == 2 ? state[1 - parity] + binding.offset : scratch.floats() + binding.offset;
                require(reinterpret_cast<uintptr_t>(pointer) % 256 == 0, "TensorRT I/O pointer is not 256-byte aligned");
                require(target.context[parity]->setTensorAddress(binding.name.c_str(), pointer), "TensorRT tensor address binding failed: " + binding.name);
            }
        }
    }

    void pipeline(unsigned p) {
        check(cudaMemcpyAsync(scratch.floats() + input_offset, host_input, FRAME * sizeof(float), cudaMemcpyHostToDevice, stream), "frame H2D");
        for (size_t i = 0; i < dispatches.size();) {
            if (dispatches[i].shader == 0) {
                unsigned which = i == ranges[0].first ? 0 : 1;
                require(engines[which].context[p]->enqueueV3(stream), "TensorRT enqueueV3 failed\n" + logger.snapshot());
                i = ranges[which].second;
            } else {
                const auto& d = dispatches[i];
                auto* w = weights.floats();
                auto* in = state[p];
                auto* out = state[1 - p];
                auto* s = scratch.floats();
                auto* params = static_cast<uint32_t*>(parameters.p) + i * 32;
                void* arguments[] = {&w, &in, &out, &s, &params};
                driver(cuLaunchKernel(functions[d.shader], d.groups, 1, 1, 128, 1, 1, 0, stream, arguments, nullptr), "CUDA DSP/FFT launch");
                ++i;
            }
        }
        check(cudaMemcpyAsync(host_output, scratch.floats() + output_offset, FRAME * sizeof(float), cudaMemcpyDeviceToHost, stream), "frame D2H");
    }

    void zero() {
        for (auto* s : state) check(cudaMemsetAsync(s, 0, STATE * sizeof(float), stream), "reset state");
        check(cudaMemsetAsync(scratch.p, 0, size_t(scratch_len) * sizeof(float), stream), "reset scratch");
        check(cudaStreamSynchronize(stream), "reset completion");
        parity = 0;
    }

    void initialize(const float* packed, size_t count, uint32_t len, uint32_t input, uint32_t output,
        const AnrDispatch* schedule, size_t schedule_count, size_t attenuation_dispatch, const char* source) {
        require(schedule && schedule_count && attenuation_dispatch < schedule_count && source, "invalid packed model ABI");
        require(input + FRAME <= len && output + FRAME <= len, "frame scratch offsets out of bounds");
        scratch_len = len; input_offset = input; output_offset = output; attenuation_index = attenuation_dispatch;
        dispatches.assign(schedule, schedule + schedule_count);
        // cudaFree(nullptr) establishes the runtime primary context used by TRT
        // and the CUDA Driver module. Nothing is selected on a CPU device.
        check(cudaSetDevice(0), "cudaSetDevice(0): install a compatible NVIDIA driver and CUDA GPU");
        check(cudaFree(nullptr), "CUDA primary context initialization");
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
        weights.allocate(count * sizeof(float));
        scratch.allocate(size_t(len) * sizeof(float));
        parameters.allocate(schedule_count * 32 * sizeof(uint32_t));
        check(cudaMemcpyAsync(weights.p, packed, count * sizeof(float), cudaMemcpyHostToDevice, stream), "weight upload");
        std::vector<uint32_t> params;
        params.reserve(schedule_count * 32);
        for (const auto& d : dispatches) params.insert(params.end(), d.p, d.p + 32);
        check(cudaMemcpyAsync(parameters.p, params.data(), params.size() * sizeof(uint32_t), cudaMemcpyHostToDevice, stream), "parameter upload");
        for (unsigned p = 0; p < 2; ++p) {
            state_allocation[p].allocate(STATE * sizeof(float) + 256);
            state[p] = state_allocation[p].floats() + 8; // +32 bytes aligns existing five GRU offsets.
        }
        check(cudaMallocHost(reinterpret_cast<void**>(&host_input), FRAME * sizeof(float)), "pinned frame input allocation");
        check(cudaMallocHost(reinterpret_cast<void**>(&host_output), FRAME * sizeof(float)), "pinned frame output allocation");
        std::fill_n(host_input, FRAME, 0.0f);
        check(cudaStreamSynchronize(stream), "model upload completion");
        compile_cuda(source);
        runtime.reset(checked(createInferRuntime(logger), "TensorRT runtime creation failed"));
        unsigned ranges_found = 0;
        std::map<uint32_t, std::set<size_t>> consumers;
        for (size_t i = 0; i < dispatches.size(); ++i) {
            const auto& d = dispatches[i];
            require(d.shader <= 2, "unknown ShaderKind ABI");
            if (d.shader == 0) {
                if (i == 0 || dispatches[i - 1].shader != 0) {
                    require(ranges_found < 2, "expected two contiguous neural ranges");
                    ranges[ranges_found++] = {i, i};
                }
                ranges[ranges_found - 1].second = i + 1;
                for (auto r : reads(d)) consumers[r.first].insert(i);
            } else if (d.shader == 1 && d.p[0] == 3) consumers[d.p[12]].insert(i);
            else if (d.shader == 1 && d.p[0] == 4) {
                consumers[d.p[13]].insert(i);
                consumers[d.p[14]].insert(i);
            }
        }
        require(ranges_found == 2 && ranges[0].second + 1 == ranges[1].first &&
            dispatches[ranges[0].second].shader == 1 && dispatches[ranges[0].second].p[0] == 3,
            "DFN3 requires encoder -> CUDA DSP history mode3 -> decoder");
        build_engine(0, packed, count, consumers);
        build_engine(1, packed, count, consumers);
        zero();
        // TRT can lazily configure tactics on its first enqueue. Warm both
        // parity contexts outside capture, then capture the complete frame.
        for (unsigned p = 0; p < 2; ++p) {
            pipeline(p);
            check(cudaStreamSynchronize(stream), "TensorRT warmup completion");
        }
        zero();
        for (unsigned p = 0; p < 2; ++p) {
            check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin frame graph capture");
            try {
                pipeline(p);
                check(cudaStreamEndCapture(stream, &graph[p]), "end frame graph capture");
            } catch (...) {
                cudaGraph_t abandoned = nullptr;
                cudaStreamEndCapture(stream, &abandoned);
                if (abandoned) cudaGraphDestroy(abandoned);
                throw;
            }
            check(cudaGraphInstantiate(&executable[p], graph[p], 0), "instantiate frame graph");
        }
        zero();
    }

    void frame(const float* input, float* output, float db, bool reset) {
        require(!failed, "CUDA/TensorRT stream previously failed; recreate the reducer");
        try {
            if (reset) zero();
            if (attenuation != db) {
                auto& d = dispatches[attenuation_index];
                float alpha = std::pow(10.0f, -std::abs(db) / 20.0f);
                std::memcpy(&d.p[22], &alpha, sizeof(alpha));
                d.p[23] = std::abs(db) > 0 ? 1 : 0;
                check(cudaMemcpyAsync(static_cast<uint32_t*>(parameters.p) + attenuation_index * 32 + 22,
                    d.p + 22, 2 * sizeof(uint32_t), cudaMemcpyHostToDevice, stream), "attenuation parameter upload");
                attenuation = db;
            }
            std::copy_n(input, FRAME, host_input);
            check(cudaGraphLaunch(executable[parity], stream), "frame graph replay");
            check(cudaStreamSynchronize(stream), "frame graph completion");
            std::copy_n(host_output, FRAME, output);
            parity = 1 - parity;
        } catch (...) {
            failed = true;
            if (stream) cudaStreamSynchronize(stream);
            throw;
        }
    }
};
} // namespace

extern "C" void* anr_trt_create(const float* weights, size_t weight_count, uint32_t scratch_len,
    uint32_t input, uint32_t output, const AnrDispatch* dispatches, size_t count,
    size_t attenuation_dispatch, const char* source, char* error, size_t capacity) noexcept {
    try {
        auto instance = std::make_unique<Instance>();
        try {
            instance->initialize(weights, weight_count, scratch_len, input, output, dispatches,
                count, attenuation_dispatch, source);
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string(e.what()) + "\nTensorRT logger:\n" + instance->logger.snapshot());
        }
        return instance.release();
    } catch (const std::exception& e) { report(error, capacity, e.what()); }
    catch (...) { report(error, capacity, "unknown native exception creating TensorRT reducer"); }
    return nullptr;
}
extern "C" int anr_trt_frame(void* instance, const float* input, float* output,
    float attenuation, int reset, char* error, size_t capacity) noexcept {
    try {
        require(instance && input && output, "invalid TensorRT frame ABI pointers");
        static_cast<Instance*>(instance)->frame(input, output, attenuation, reset != 0);
        return 0;
    } catch (const std::exception& e) { report(error, capacity, e.what()); }
    catch (...) { report(error, capacity, "unknown native exception executing TensorRT frame"); }
    return -1;
}
extern "C" void anr_trt_destroy(void* instance) noexcept {
    delete static_cast<Instance*>(instance);
}
extern "C" const char* anr_trt_last_error() noexcept {
    return last_error.c_str();
}
