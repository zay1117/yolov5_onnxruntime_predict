#include "utils.h"

// 计算整数向量的乘积
size_t utils::vectorProduct(const std::vector<int64_t>& vector)
{
    if (vector.empty())
        return 0; // 如果向量为空，返回0

    size_t product = 1; // 初始乘积为1
    for (const auto& element : vector)
        product *= element; // 计算乘积

    return product;
}

// 将字符型字符串转换为宽字符型字符串
std::wstring utils::charToWstring(const char* str)
{
    typedef std::codecvt_utf8<wchar_t> convert_type; // 定义字符转换类型
    std::wstring_convert<convert_type, wchar_t> converter; // 创建转换器

    return converter.from_bytes(str); // 执行转换
}

// 从文件中加载类名
std::vector<std::string> utils::loadNames(const std::string& path)
{
    std::vector<std::string> classNames; // 用于存储类名的向量
    std::ifstream infile(path); // 打开文件
    if (infile.good()) // 检查文件是否成功打开
    {
        std::string line;
        while (getline (infile, line)) // 逐行读取文件
        {
            if (line.back() == '\r') // 去除Windows风格的回车符
                line.pop_back();
            classNames.emplace_back(line); // 将每行内容添加到类名向量中
        }
        infile.close(); // 关闭文件
    }
    else
    {
        std::cerr << "ERROR: Failed to access class name path: " << path << std::endl; // 输出错误信息
    }
    return classNames;
}

// 可视化检测结果
void utils::visualizeDetection(cv::Mat& image, std::vector<Detection>& detections,
                               const std::vector<std::string>& classNames)
{
    for (const Detection& detection : detections) // 遍历所有检测结果
    {
        cv::rectangle(image, detection.box, cv::Scalar(229, 160, 21), 2); // 绘制检测框

        int x = detection.box.x;
        int y = detection.box.y;

        int conf = (int)std::round(detection.conf * 100); // 将置信度转换为百分比
        int classId = detection.classId; // 获取类别ID
        std::string label = classNames[classId] + " 0." + std::to_string(conf); // 创建标签字符串

        int baseline = 0;
        cv::Size size = cv::getTextSize(label, cv::FONT_ITALIC, 0.8, 2, &baseline); // 获取文本尺寸
        cv::rectangle(image,
                      cv::Point(x, y - 25), cv::Point(x + size.width, y),
                      cv::Scalar(229, 160, 21), -1); // 绘制背景矩形

        cv::putText(image, label,
                    cv::Point(x, y - 3), cv::FONT_ITALIC,
                    0.8, cv::Scalar(255, 255, 255), 2); // 绘制文本
    }
}

// 将图像调整为指定大小并填充
void utils::letterbox(const cv::Mat& image, cv::Mat& outImage,
                      const cv::Size& newShape = cv::Size(640, 640),
                      const cv::Scalar& color = cv::Scalar(114, 114, 114),
                      bool auto_ = true,
                      bool scaleFill = false,
                      bool scaleUp = true,
                      int stride = 32)
{
    cv::Size shape = image.size(); // 获取原图像尺寸
    float r = std::min((float)newShape.height / (float)shape.height,
                       (float)newShape.width / (float)shape.width); // 计算缩放比例
    if (!scaleUp)
        r = std::min(r, 1.0f); // 如果不允许放大，则将比例限制为1.0

    float ratio[2] {r, r}; // 缩放比例
    int newUnpad[2] {(int)std::round((float)shape.width * r),
                     (int)std::round((float)shape.height * r)}; // 缩放后的尺寸

    auto dw = (float)(newShape.width - newUnpad[0]); // 计算宽度填充
    auto dh = (float)(newShape.height - newUnpad[1]); // 计算高度填充

    if (auto_)
    {
        dw = (float)((int)dw % stride); // 自动调整填充
        dh = (float)((int)dh % stride);
    }
    else if (scaleFill)
    {
        dw = 0.0f;
        dh = 0.0f;
        newUnpad[0] = newShape.width; // 填充模式：调整图像到目标尺寸
        newUnpad[1] = newShape.height;
        ratio[0] = (float)newShape.width / (float)shape.width;
        ratio[1] = (float)newShape.height / (float)shape.height;
    }

    dw /= 2.0f;
    dh /= 2.0f;

    if (shape.width != newUnpad[0] && shape.height != newUnpad[1])
    {
        cv::resize(image, outImage, cv::Size(newUnpad[0], newUnpad[1])); // 调整图像尺寸
    }

    int top = int(std::round(dh - 0.1f)); // 计算填充边界
    int bottom = int(std::round(dh + 0.1f));
    int left = int(std::round(dw - 0.1f));
    int right = int(std::round(dw + 0.1f));
    cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color); // 填充边界
}

// 根据新的图像尺寸调整坐标
void utils::scaleCoords(const cv::Size& imageShape, cv::Rect& coords, const cv::Size& imageOriginalShape)
{
    float gain = std::min((float)imageShape.height / (float)imageOriginalShape.height,
                          (float)imageShape.width / (float)imageOriginalShape.width); // 计算缩放因子

    int pad[2] {(int) (( (float)imageShape.width - (float)imageOriginalShape.width * gain) / 2.0f),
                (int) (( (float)imageShape.height - (float)imageOriginalShape.height * gain) / 2.0f)}; // 计算填充量

    coords.x = (int) std::round(((float)(coords.x - pad[0]) / gain)); // 缩放坐标
    coords.y = (int) std::round(((float)(coords.y - pad[1]) / gain));

    coords.width = (int) std::round(((float)coords.width / gain)); // 缩放宽度和高度
    coords.height = (int) std::round(((float)coords.height / gain));
}

// 将值限制在指定范围内
template <typename T>
T utils::clip(const T& n, const T& lower, const T& upper)
{
    return std::max(lower, std::min(n, upper)); // 限制值在下界和上界之间
}