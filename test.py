import torch
import torchvision.models as models
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time
import os

# TensorRT 로거 설정
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_tensorrt_engine(onnx_file_path):
    """ONNX 파일을 파싱하여 TensorRT 엔진을 빌드합니다."""
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # TensorRT 컴파일 중 사용할 최대 작업 공간 설정 (1GB)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise RuntimeError("ONNX 파싱에 실패했습니다.")
            
    engine_bytes = builder.build_serialized_network(network, config)
    runtime = trt.Runtime(TRT_LOGGER)
    return runtime.deserialize_cuda_engine(engine_bytes)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != 'cuda':
        raise SystemError("CUDA 환경(NVIDIA GPU)이 필수적으로 요구됩니다.")

    # 1. ResNet-50 모델 로드 및 가중치 초기화
    model = models.resnet50(weights=None).to(device).eval()
    
    # 배치 크기를 8로 설정하여 GPU 연산 부하 증가 (배치 사이즈 8, 3채널, 224x224 해상도)
    batch_size = 8
    dummy_input = torch.randn(batch_size, 3, 224, 224, device=device)
    onnx_file = "resnet50_model.onnx"

    print("PyTorch (ResNet-50) 추론 및 시간 측정 중...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
            
        torch.cuda.synchronize()
        start_time = time.time()
        pt_output = model(dummy_input)
        torch.cuda.synchronize()
        pt_latency = time.time() - start_time
        pt_output_np = pt_output.cpu().numpy()

    print("ONNX 익스포트 및 TensorRT 엔진 빌드 중 (모델이 커서 빌드에 시간이 다소 소요됩니다)...")
    # 2. 에러 해결: opset_version을 18로 상향 조정
    torch.onnx.export(model, dummy_input, onnx_file, 
                      input_names=['input'], output_names=['output'], 
                      opset_version=18)

    engine = build_tensorrt_engine(onnx_file)
    context = engine.create_execution_context()

    # 3. 메모리 할당
    input_data = dummy_input.cpu().numpy().astype(np.float32)
    h_output = np.empty(pt_output_np.shape, dtype=np.float32)
    
    d_input = cuda.mem_alloc(input_data.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    bindings = [int(d_input), int(d_output)]

    print("TensorRT (ResNet-50) 추론 및 시간 측정 중...")
    for _ in range(10):
        cuda.memcpy_htod(d_input, input_data)
        context.execute_v2(bindings)
        cuda.memcpy_dtoh(h_output, d_output)
        cuda.Context.synchronize()

    start_time = time.time()
    cuda.memcpy_htod(d_input, input_data)
    context.execute_v2(bindings)
    cuda.memcpy_dtoh(h_output, d_output)
    cuda.Context.synchronize()
    trt_latency = time.time() - start_time

    max_diff = np.max(np.abs(pt_output_np - h_output))
    
    print("\n=== ResNet-50 실행 결과 비교 ===")
    print(f"PyTorch 지연 시간: {pt_latency * 1000:.4f} ms")
    print(f"TensorRT 지연 시간: {trt_latency * 1000:.4f} ms")
    print(f"성능 향상 배수: {pt_latency / trt_latency:.2f}배")
    print(f"부동소수점 오차 (Max Absolute Error): {max_diff:.8e}")
    
    if os.path.exists(onnx_file):
        os.remove(onnx_file)