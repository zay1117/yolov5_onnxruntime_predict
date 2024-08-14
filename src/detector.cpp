#include "detector.h"
#include <typeinfo>

YOLODetector::YOLODetector(const std::string& modelPath, //模型路径
                           const bool& isGPU = true,  //是否使用GPU
                           const cv::Size& inputSize = cv::Size(640, 640))  //输入图像大小
{
    env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "ONNX_DETECTION");  //初始化onnx runtime环境,设置日志级别为警告
    sessionOptions = Ort::SessionOptions();  //创建一个InferenceSession对象来加载模型

    //获得支持的执行提供者列表
    std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
    auto cudaAvailable = std::find(availableProviders.begin(), availableProviders.end(), "CUDAExecutionProvider");
    //创建CUDA提供者选项
    OrtCUDAProviderOptions cudaOption;
    //判断是否使用GPU，并检查是否支持CUDA
    if (isGPU && (cudaAvailable == availableProviders.end()))
    {
        std::cout << "GPU is not supported by your ONNXRuntime build. Fallback to CPU." << std::endl;
        std::cout << "Inference device: CPU" << std::endl;
    }
    else if (isGPU && (cudaAvailable != availableProviders.end()))
    {
        std::cout << "Inference device: GPU" << std::endl;
        //添加CUDA执行提供者
        sessionOptions.AppendExecutionProvider_CUDA(cudaOption);
    }
    else
    {
        std::cout << "Inference device: CPU" << std::endl;
    }

    //创建ONNX runtime 并加载模型
    session = Ort::Session(env, modelPath.c_str(), sessionOptions);

    //创建默认的内存分配器
    Ort::AllocatorWithDefaultOptions allocator;
    //获取输入张量的类型信息
    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    //获取输入张量的形状
    std::vector<int64_t> inputTensorShape = inputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
    this->isDynamicInputShape = false;
    // 检查输入的张量的宽度和高度是否为动态大小
    if (inputTensorShape[2] == -1 && inputTensorShape[3] == -1)
    {
        std::cout << "Dynamic input shape" << std::endl;
        this->isDynamicInputShape = true;
    }
    //打印输入张量的形状
    for (auto shape : inputTensorShape)
        std::cout << "Input shape: " << shape << std::endl;

    inputNames.push_back(session.GetInputName(0, allocator));
    outputNames.push_back(session.GetOutputName(0, allocator));

    std::cout << "Input name: " << inputNames[0] << std::endl;
    std::cout << "Output name: " << outputNames[0] << std::endl;
    // 设置输入图像的大小
    this->inputImageShape = cv::Size2f(inputSize);
}

//获取检测到的最佳类别信息
void YOLODetector::getBestClassInfo(std::vector<float>::iterator it, const int& numClasses,
                                    float& bestConf, int& bestClassId)
{
    // 前五个元素是框和对象置信度
    bestClassId = 5;
    bestConf = 0;

    // 遍历所有类别找到置信度最高的类别
    for (int i = 5; i < numClasses + 5; i++)
    {
        if (it[i] > bestConf)
        {
            bestConf = it[i];
            bestClassId = i - 5;
        }
    }

}

// 对输入的图像进行预处理
void YOLODetector::preprocessing(cv::Mat &image, float*& blob, std::vector<int64_t>& inputTensorShape)
{
    cv::Mat resizedImage, floatImage;
    //将图像从BRG转换为RGB
    cv::cvtColor(image, resizedImage, cv::COLOR_BGR2RGB);
    // 调整图像大小，并填充边框，保持长宽比
    utils::letterbox(resizedImage, resizedImage, this->inputImageShape,
                     cv::Scalar(114, 114, 114), this->isDynamicInputShape,
                     false, true, 32);
    // 更新输入张量的形状
    inputTensorShape[2] = resizedImage.rows;
    inputTensorShape[3] = resizedImage.cols;
    // 将图像像素值归一化到【0,1】的范围
    resizedImage.convertTo(floatImage, CV_32FC3, 1 / 255.0);
    // 创建一个新的FLOAT数组作为输入张量的内存
    blob = new float[floatImage.cols * floatImage.rows * floatImage.channels()];
    cv::Size floatImageSize {floatImage.cols, floatImage.rows};

    // hwc -> chw，图像格式转换
    std::vector<cv::Mat> chw(floatImage.channels());
    for (int i = 0; i < floatImage.channels(); ++i)
    {
        chw[i] = cv::Mat(floatImageSize, CV_32FC1, blob + i * floatImageSize.width * floatImageSize.height);
    }
    cv::split(floatImage, chw);
}

// 对模型的输入结果进行后处理
std::vector<Detection> YOLODetector::postprocessing(const cv::Size& resizedImageShape,
                                                    const cv::Size& originalImageShape,
                                                    std::vector<Ort::Value>& outputTensors,
                                                    const float& confThreshold, const float& iouThreshold)
{
    std::vector<cv::Rect> boxes;
    std::vector<float> confs;
    std::vector<int> classIds;

    auto* rawOutput = outputTensors[0].GetTensorData<float>();
    std::vector<int64_t> outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t count = outputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
    std::vector<float> output(rawOutput, rawOutput + count);

    for (const int64_t& shape : outputShape)
        std::cout << "Output Shape: " << shape << std::endl;

    // 前5个元素是框和置信度
    int numClasses = (int)outputShape[2] - 5;
    int elementsInBatch = (int)(outputShape[1] * outputShape[2]);

    // 对于批次大小为1的情况
    for (auto it = output.begin(); it != output.begin() + elementsInBatch; it += outputShape[2])
    {
        float clsConf = it[4];
        //仅在类别置信度高于阈值时进行处理
        if (clsConf > confThreshold)
        {
            int centerX = (int) (it[0]);
            int centerY = (int) (it[1]);
            int width = (int) (it[2]);
            int height = (int) (it[3]);
            int left = centerX - width / 2;
            int top = centerY - height / 2;

            float objConf;
            int classId;
            // 获取最佳类别信息
            this->getBestClassInfo(it, numClasses, objConf, classId);

            float confidence = clsConf * objConf;
            // 储存检测到的框、置信度和类别ID
            boxes.emplace_back(left, top, width, height);
            confs.emplace_back(confidence);
            classIds.emplace_back(classId);
        }
    }

    //非极大抑制（NMS）
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confs, confThreshold, iouThreshold, indices);
    // std::cout << "amount of NMS indices: " << indices.size() << std::endl;

    std::vector<Detection> detections;
    // 根据NMS创建检测结果
    for (int idx : indices)
    {
        Detection det;
        det.box = cv::Rect(boxes[idx]);
        utils::scaleCoords(resizedImageShape, det.box, originalImageShape);

        det.conf = confs[idx];
        det.classId = classIds[idx];
        detections.emplace_back(det);
    }

    return detections;
}

//执行目标检测
std::vector<Detection> YOLODetector::detect(cv::Mat &image, const float& confThreshold = 0.4,
                                            const float& iouThreshold = 0.45)
{
    float *blob = nullptr;
    //定义输入张量的形状，1表示批次大小，3表示输入图像的通道数，-1表示动态尺寸，预处理后将被更新
    std::vector<int64_t> inputTensorShape {1, 3, -1, -1};
    this->preprocessing(image, blob, inputTensorShape);  //预处理图像，生成输入张量
    // 计算输入张量的大小
    size_t inputTensorSize = utils::vectorProduct(inputTensorShape);

    std::vector<float> inputTensorValues(blob, blob + inputTensorSize);

    std::vector<Ort::Value> inputTensors;

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    inputTensors.push_back(Ort::Value::CreateTensor<float>(
            memoryInfo, inputTensorValues.data(), inputTensorSize,
            inputTensorShape.data(), inputTensorShape.size()
    ));

    std::vector<Ort::Value> outputTensors = this->session.Run(Ort::RunOptions{nullptr},
                                                              inputNames.data(),
                                                              inputTensors.data(),
                                                              1,
                                                              outputNames.data(),
                                                              1);

    cv::Size resizedShape = cv::Size((int)inputTensorShape[3], (int)inputTensorShape[2]);
    std::vector<Detection> result = this->postprocessing(resizedShape,
                                                         image.size(),
                                                         outputTensors,
                                                         confThreshold, iouThreshold);

    delete[] blob;

    return result;
}
