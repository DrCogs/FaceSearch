#define NOMINMAX
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <filesystem>
#include <ctime>
#include <chrono>
#include <thread>

//全局变量
int cap_width = 1280;
int cap_height = 720;
int roomnumber = 0;
int camera = 0;
std::string log_filename;  // 日志文件名：roomnumber + 年月日

//时间截取（跨平台，使用 <ctime>）
std::string GetTime(std::string mode)
{
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::tm local_time;
    localtime_s(&local_time, &time_t_now);

    char rettime[30];
    if (mode == "getface")
    {
        std::strftime(rettime, sizeof(rettime), "%Y-%m-%d %H:%M:%S", &local_time);
    }
    else if (mode == "Config")
    {
        std::strftime(rettime, sizeof(rettime), "%Y-%m-%d", &local_time);
    }
    return rettime;
}

//与特征提取程序保持一致的数据结构
struct FaceRecord
{
    std::string label;          // 人脸标签（人名/ID）
    std::string source_image;   // 来源图片路径
    cv::Mat     features;       // 特征向量 (1 x 128, CV_32F)
    float       confidence;     // 检测置信度
    int         face_x, face_y, face_w, face_h; // 人脸框位置
};

//配置参数
struct RecognizeConfig
{
    std::string yunet_model = "models/face_detection_yunet_2023mar.onnx";
    std::string sface_model = "models/face_recognition_sface_2021dec.onnx";

    //待识别的照片
    std::string target_image = "face.jpg";

    //特征库文件
    std::string feature_file = "face_features.dat";

    //识别阈值（cosine 相似度 >= 此值才认为是同一人）
    float match_threshold = 0.363f;

    //YuNet 的检测参数
    float detect_threshold = 0.5f;
    float nms_threshold = 0.3f;
    int top_k = 5000;

    //DNN后端
    int backend_id = cv::dnn::DNN_BACKEND_OPENCV;
    int target_id = cv::dnn::DNN_TARGET_CPU;
};

//从 .dat 文件加载特征（与提取程序格式完全一致）
bool loadFeatures(const std::string& filepath, std::vector<FaceRecord>& records)
{
    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs.is_open())
    {
        std::cerr << "[ERROR] 无法打开特征文件: " << filepath << std::endl;
        return false;
    }
    //读取魔数 + 版本号
    uint32_t magic, version;
    ifs.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    ifs.read(reinterpret_cast<char*>(&version), sizeof(version));

    if (magic != 0x46524543)
    {
        std::cerr << "[ERROR] 无效的特征文件格式" << std::endl;
        return false;
    }
    //读取记录数量
    uint32_t count;
    ifs.read(reinterpret_cast<char*>(&count), sizeof(count));

    records.resize(count);
    for (uint32_t i = 0; i < count; ++i)
    {
        //读取标签
        uint32_t label_len;
        ifs.read(reinterpret_cast<char*>(&label_len), sizeof(label_len));
        records[i].label.resize(label_len);
        ifs.read(&records[i].label[0], label_len);
        // 读取来源路径
        uint32_t src_len;
        ifs.read(reinterpret_cast<char*>(&src_len), sizeof(src_len));
        records[i].source_image.resize(src_len);
        ifs.read(&records[i].source_image[0], src_len);
        // 读取置信度
        ifs.read(reinterpret_cast<char*>(&records[i].confidence), sizeof(float));
        // 读取人脸框
        ifs.read(reinterpret_cast<char*>(&records[i].face_x), sizeof(int));
        ifs.read(reinterpret_cast<char*>(&records[i].face_y), sizeof(int));
        ifs.read(reinterpret_cast<char*>(&records[i].face_w), sizeof(int));
        ifs.read(reinterpret_cast<char*>(&records[i].face_h), sizeof(int));
        // 读取特征向量
        int rows, cols, type;
        ifs.read(reinterpret_cast<char*>(&rows), sizeof(int));
        ifs.read(reinterpret_cast<char*>(&cols), sizeof(int));
        ifs.read(reinterpret_cast<char*>(&type), sizeof(int));
        records[i].features.create(rows, cols, type);
        ifs.read(reinterpret_cast<char*>(records[i].features.data),
            records[i].features.total() * records[i].features.elemSize());
    }

    ifs.close();
    std::cout << "[INFO] 已加载 " << records.size() << " 条人脸特征记录" << std::endl;
    return true;
}

//核心类
class FaceRecognizer
{
public:
    FaceRecognizer(const RecognizeConfig& config)
        : config_(config)
    {
        //初始化YuNet
        detector_ = cv::FaceDetectorYN::create(
            config_.yunet_model, "",
            cv::Size(320, 320),
            config_.detect_threshold,
            config_.nms_threshold,
            config_.top_k,
            config_.backend_id,
            config_.target_id
        );
        // 初始化SFace
        recognizer_ = cv::FaceRecognizerSF::create(
            config_.sface_model, "",
            config_.backend_id,
            config_.target_id
        );

        std::cout << "[INFO] 模型加载完成" << std::endl;
    }
    /*
     识别单张图片中的人脸
     @param image_path 待识别的图片路径
     @param database 人脸特征数据库
     @return 识别结果（人名），未识别返回 "unknown"
     */
    std::string recognize(const std::string& image_path, const std::vector<FaceRecord>& database)
    {
        //读取待识别图片
        cv::Mat image = cv::imread(image_path);
        if (image.empty())
        {
            std::cerr << "[ERROR] 无法读取图片: " << image_path << std::endl;
            return "unknown";
        }

        std::cout << "[INFO] 正在识别: " << image_path << std::endl;

        //检测人脸
        detector_->setInputSize(image.size());
        cv::Mat faces;
        detector_->detect(image, faces);

        if (faces.empty())
        {
            std::cerr << "[WARN] 图片中未检测到人脸" << std::endl;
            return "unknown";
        }

        std::cout << "[INFO] 检测到 " << faces.rows << " 张人脸，开始比对..." << std::endl;

        //取置信度最高的人脸进行识别（如果有多张人脸）
        int best_face_idx = 0;
        float best_face_conf = faces.at<float>(0, 14);
        for (int i = 1; i < faces.rows; ++i)
        {
            float conf = faces.at<float>(i, 14);
            if (conf > best_face_conf)
            {
                best_face_conf = conf;
                best_face_idx = i;
            }
        }

        //提取待识别人脸的特征
        cv::Mat target_face = faces.row(best_face_idx);
        cv::Mat aligned_face;
        recognizer_->alignCrop(image, target_face, aligned_face);

        cv::Mat target_features;
        recognizer_->feature(aligned_face, target_features);

        //与数据库中所有人脸比对
        std::string best_match = "unknown";
        double best_score = -1.0;
        std::map<std::string, double> label_best_scores; //记录每个人的最高分

        for (const auto& record : database)
        {
            double score = recognizer_->match(
                target_features,
                record.features,
                cv::FaceRecognizerSF::DisType::FR_COSINE
            );

            //更新该标签的最高分
            if (label_best_scores.find(record.label) == label_best_scores.end() ||
                score > label_best_scores[record.label])
            {
                label_best_scores[record.label] = score;
            }
        }

        //找出分数最高的人
        for (const auto& [label, score] : label_best_scores)
        {
            std::cout << "       与 [" << label << "] 的相似度: "
                << std::fixed << std::setprecision(4) << score << std::endl;

            if (score > best_score)
            {
                best_score = score;
                best_match = label;
            }
        }

        //判断是否超过阈值
        if (best_score >= config_.match_threshold)
        {
            std::cout << "\n[RESULT] 识别成功!" << std::endl;
            std::cout << "         身份: " << best_match << std::endl;
            std::cout << "         相似度: " << std::fixed << std::setprecision(4) << best_score << std::endl;
        }
        else
        {
            std::cout << "\n[RESULT] 识别失败 - 未找到匹配人员" << std::endl;
            std::cout << "         最高分: " << best_match << " (" << best_score << ")" << std::endl;
            std::cout << "         阈值: " << config_.match_threshold << "，未达到识别标准" << std::endl;
            best_match = "unknown";
        }

        return best_match;
    }

private:
    RecognizeConfig config_;
    cv::Ptr<cv::FaceDetectorYN> detector_;
    cv::Ptr<cv::FaceRecognizerSF> recognizer_;
};

//起调主函数
void gettingface()
{
    std::cout << "============================================================" << std::endl;
    std::cout << "  人脸识别程序" << std::endl;
    std::cout << "============================================================" << std::endl;

    //配置参数
    RecognizeConfig config;

    //检查模型文件是否存在
    if (!std::filesystem::exists(config.yunet_model))
    {
        std::cerr << "[ERROR] YuNet 模型不存在: " << config.yunet_model << std::endl;
        return;
    }
    if (!std::filesystem::exists(config.sface_model))
    {
        std::cerr << "[ERROR] SFace 模型不存在: " << config.sface_model << std::endl;
        return;
    }

    //检查待识别图片是否存在
    if (!std::filesystem::exists(config.target_image))
    {
        std::cerr << "[ERROR] 待识别图片不存在: " << config.target_image << std::endl;
        std::cerr << "       请将待识别的人脸照片命名为 'face.jpg' 放到程序目录" << std::endl;
        return;
    }

    //检查特征文件是否存在
    if (!std::filesystem::exists(config.feature_file))
    {
        std::cerr << "[ERROR] 特征文件不存在: " << config.feature_file << std::endl;
        std::cerr << "       请先运行特征提取程序生成 face_features.dat" << std::endl;
        return;
    }

    //加载特征数据库
    std::vector<FaceRecord> database;
    if (!loadFeatures(config.feature_file, database))
    {
        return;
    }

    //创建识别器并执行识别
    FaceRecognizer recognizer(config);
    std::string result = recognizer.recognize(config.target_image, database);

    std::cout << "\n============================================================" << std::endl;
    std::cout << "  最终结果: " << result << std::endl;
    std::cout << "============================================================" << std::endl;

    //输入日志（追加模式，每条一行）
    std::fstream outputing;
    outputing.open(log_filename, std::ios::out | std::ios::app);
    if (!outputing || result == "unknown")
    {
        std::cout << "[ERROR] 无法打开日志文件 或 不是对应人物 " << log_filename << std::endl;
    }
    else
    {
        outputing << roomnumber << " | " << result << " | " << GetTime("getface") << std::endl;
        outputing.close();
        std::cout << "[INFO] 日志已写入: " << log_filename << std::endl;
    }
    std::filesystem::remove("face.jpg");
}

//拍照
void CaptureFace()
{
    std::filesystem::remove("face.jpg");
    cv::VideoCapture Picture(camera);
    Picture.set(cv::CAP_PROP_FRAME_WIDTH, cap_width);
    Picture.set(cv::CAP_PROP_FRAME_HEIGHT, cap_height);

    cv::Mat frame;
    Picture >> frame;

    cv::imwrite("face.jpg", frame);
    Picture.release();
}

//加载数据
void LoadConfig()
{
    std::ifstream loadconfig("Config.txt");
    std::string line;
    while (std::getline(loadconfig, line))
    {
        int loads = line.find(':');
        if (loads < 0) continue;
        //左名，右值
        std::string name = line.substr(0, loads);
        std::string number = line.substr(loads + 1);
        //匹配
        if (name == "cap_width") cap_width = std::stoi(number);
        if (name == "cap_height") cap_height = std::stoi(number);
        if (name == "roomnumber") roomnumber = std::stoi(number);
        if (name == "camera") camera = std::stoi(number);
    }
}

//初始化日志文件（程序启动时调用，生成 roomnumber + 年月日 格式的文件名）
void InitLogFile()
{
    // 生成文件名：roomnumber_年月日.txt
    log_filename = std::to_string(roomnumber) + "_" + GetTime("Config") + ".txt";

    // 检查文件是否存在，不存在创建
    if (!std::filesystem::exists(log_filename))
    {
        std::ofstream create_file(log_filename);
        if (create_file.is_open())
        {
            create_file << "            打卡签到日志" << std::endl;
            create_file << "=========================" << std::endl;
            create_file.close();
            std::cout << "[INFO] 创建新日志文件: " << log_filename << std::endl;
        }
        else
        {
            std::cerr << "[ERROR] 无法创建日志文件: " << log_filename << std::endl;
        }
    }
    else
    {
        std::cout << "[INFO] 使用已有日志文件: " << log_filename << std::endl;
    }
}

//Main
int main(int argc, char** argv)
{
    LoadConfig();
    InitLogFile();   // 初始化日志文件（必须在 LoadConfig 之后，因为需要 roomnumber）
    CaptureFace();
    gettingface();
    std::this_thread::sleep_for(std::chrono::seconds(10));
    return 0;
}