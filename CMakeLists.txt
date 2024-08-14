cmake_minimum_required(VERSION 3.10)
project(yolov5_onnxruntime_predict)

set(CMAKE_CXX_STANDARD 14)

# 设置ONNX Runtime路径，这里换成自己的 onnxruntime,推荐版本是1.12
set(ONNXRUNTIME_DIR "/home/zhangao/software/onnxruntime-linux-x64-gpu-1.12.0/")
include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

# 设置CUDA路径
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

# 设置OpenCV路径
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 添加头文件，头文件位于include中
include_directories("/home/zhangao/code/yolov5_onnxruntime_predict/include")

# 添加可执行文件及其源文件
add_executable(yolov5_onnxruntime_predict
               src/main.cpp
               src/detector.cpp
               src/utils.cpp)

# 链接OpenCV、ONNX Runtime和CUDA库,主要是libonnxruntime.so.1.12.0和libonnxruntime_providers_cuda.so这两个动态库要添加进来
target_link_libraries(yolov5_onnxruntime_predict ${OpenCV_LIBS} ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so.1.12.0 ${ONNXRUNTIME_DIR}/lib/libonnxruntime_providers_cuda.so ${CUDA_LIBRARIES})




