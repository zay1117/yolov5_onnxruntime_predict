#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <chrono>

#include "utils.h"
#include "detector.h"

// 处理图像的函数，输入单张图像进行预测并输出预测后的图像
void processImage(YOLODetector& detector,
                  const std::string& imagePath,
                  float confThreshold,
                  float iouThreshold,
                  const std::vector<std::string>& classNames,
                  const std::string save_path) {
    cv::Mat image = cv::imread(imagePath);  // 读取图像
    if (image.empty()) {
        std::cerr << "错误：无法打开图像 " << imagePath << std::endl;
        return;
    }

    auto result = detector.detect(image, confThreshold, iouThreshold);  // 进行目标检测
    utils::visualizeDetection(image, result, classNames);  // 可视化检测结果

    cv::imwrite(save_path, image);  // 保存结果图像
    std::cout << "结果图像已保存为 " << save_path << std::endl;

    cv::imshow("结果", image);  // 显示结果图像
    cv::waitKey(0);  // 等待按键事件
}

// 处理摄像头输入的函数
void processCamera(YOLODetector& detector,
                   float confThreshold,
                   float iouThreshold,
                   const std::vector<std::string>& classNames) {
    cv::VideoCapture cap(0);  // 打开默认摄像头
    if (!cap.isOpened()) {
        std::cerr << "错误：无法打开摄像头。" << std::endl;
        return;
    }

    cv::Mat frame;
    int frameCount = 0;  // 帧计数器
    auto startTime = std::chrono::steady_clock::now();  // 记录开始时间

    while (true) {
        cap >> frame;  // 捕捉帧
        if (frame.empty()) {
            std::cerr << "错误：捕捉到空帧。" << std::endl;
            break;
        }

        auto result = detector.detect(frame, confThreshold, iouThreshold);  // 进行目标检测
        utils::visualizeDetection(frame, result, classNames);  // 可视化检测结果

        cv::imshow("实时检测", frame);  // 显示实时检测结果

        frameCount++;  // 增加帧计数器

        // 每秒计算一次 FPS
        auto endTime = std::chrono::steady_clock::now();
        std::chrono::duration<float> elapsed = endTime - startTime;
        if (elapsed.count() >= 1.0) {  // 每秒计算一次
            float fps = frameCount / elapsed.count();  // 计算 FPS
            std::cout << "FPS: " << fps << std::endl;
            frameCount = 0;  // 重置帧计数器
            startTime = std::chrono::steady_clock::now();  // 重新记录开始时间
        }

        if (cv::waitKey(1) == 27) {  // 按 'ESC' 键退出循环
            break;
        }
    }
    cap.release();  // 释放摄像头资源
}

int main()
{
    const float confThreshold = 0.3f;  // 置信度阈值
    const float iouThreshold = 0.4f;  // IOU 阈值
    const std::string classNamesPath = "/home/zhangao/code/yolov5_onnxruntime_predict/models/coco.names";  // 类别名称文件路径
    const std::string modelPath = "/home/zhangao/code/yolov5_onnxruntime_predict/models/yolov5n.onnx";  // 模型文件路径（onnx）
    const std::string imagePath = "/home/zhangao/code/yolov5_onnxruntime_predict/images/street.jpg";  // 单张图像预测时输入图像路径
    const std::string save_path = "/home/zhangao/code/yolov5_onnxruntime_predict/images/result_street.jpg";  // 单张图像预测时结果图像保存路径
    const bool isGPU =  true;  // 是否使用 GPU

    const std::vector<std::string> classNames = utils::loadNames(classNamesPath);  // 加载类别名称

    YOLODetector detector {nullptr};
    try
    {
        detector = YOLODetector(modelPath, isGPU, cv::Size(640, 640));  // 初始化检测器
        std::cout << "模型已初始化。" << std::endl;

        // 选择模式：image（图像）或 video（视频）
        bool useCamera = false;  // 设置为 true 以启用摄像头模式，false 以处理图像

        if (useCamera) {
            processCamera(detector, confThreshold, iouThreshold, classNames);  // 处理摄像头输入
        } else {
            processImage(detector, imagePath, confThreshold, iouThreshold, classNames, save_path);  // 处理静态图像
        }
    }
    catch(const std::exception& e)
    {
        std::cerr << "错误： " << e.what() << std::endl;  // 捕捉并显示异常
        return -1;
    }

    return 0;
}