#pragma comment(linker, "/subsystem:\"windows\" /entry:\"mainCRTStartup\"")
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <map>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <thread>
#include <chrono>
#include <ctime>
#include <opencv2/dnn_superres.hpp>
#include <tinyfiledialogs/tinyfiledialogs.h>
#include <seeta/FaceDetector.h>
#include <seeta/FaceLandmarker.h>
#include <seeta/FaceRecognizer.h>
#include <nlohmann/json.hpp>
// 全局变量
std::string g_detectorModel = "models/face_detector.csta";
std::string g_landmarkerModel = "models/face_landmarker_pts5.csta";
std::string g_recognizerModel = "models/face_recognizer.csta";
std::string g_dbPath = "face_features.db";
int g_featureSize = 0;  // 由模型自动检测，不硬编码维度
int cameraId = 0;   // 摄像机号码
int cap_width_set = 1200;
int cap_height_set = 800;
// 特征库中的单条记录
struct PersonEntry {
    std::string name;               // 人名
    std::vector<float> features;    // 特征向量（L2 归一化后）
};
std::vector<PersonEntry> g_database;  // 内存中的特征库
// 二进制数据库文件头
constexpr uint32_t DB_MAGIC = 0x53464644;  // 魔数 "SFDD"
constexpr uint32_t DB_VERSION = 1;           // 版本号
// 识别阈值：兼顾召回率与误识别率
double RECOGNITION_THRESHOLD = 0.7;   // 余弦相似度最低阈值
double MARGIN_THRESHOLD = 0.15;   // 第一名与第二名最小差距
float  DETECT_THRESHOLD_HI = 0.90f;  // 检测阈值（主）
float  DETECT_THRESHOLD_LO = 0.70f;  // 检测阈值（降级）
// 加载特征库
bool loadDatabase(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs)
    {
        tinyfd_messageBox("错误", ("无法打开特征库文件: " + path).c_str(), "ok", "error", 1);
        return false;
    }
    uint32_t magic = 0, version = 0, count = 0;
    int fileFeatureSize = 0;
    // 校验魔数
    ifs.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != DB_MAGIC)
    {
        tinyfd_messageBox("错误", "特征库文件魔数不匹配，文件损坏", "ok", "error", 1);
        return false;
    }
    // 校验版本
    ifs.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version != DB_VERSION)
    {
        tinyfd_messageBox("错误", "特征库版本不兼容", "ok", "error", 1);
        return false;
    }
    // 读取特征维度（防止异常大值导致 OOM）
    ifs.read(reinterpret_cast<char*>(&fileFeatureSize), sizeof(fileFeatureSize));
    if (fileFeatureSize <= 0 || fileFeatureSize > 100000)
    {
        tinyfd_messageBox("错误", "特征库维度数据异常", "ok", "error", 1);
        return false;
    }
    // 首次加载时记录维度，后续加载时校验一致性
    if (g_featureSize == 0) {
        g_featureSize = fileFeatureSize;
    }
    else if (g_featureSize != fileFeatureSize)
    {
        tinyfd_messageBox("错误", "特征库维度不匹配", "ok", "error", 1);
        return false;
    }
    // 读取记录数（防止异常大值导致 OOM）
    ifs.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (count > 1000000)
    {
        tinyfd_messageBox("错误", "特征库记录数量异常过大", "ok", "error", 1);
        return false;
    }
    g_database.clear();
    g_database.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t nameLen = 0;
        ifs.read(reinterpret_cast<char*>(&nameLen), sizeof(nameLen));
        if (nameLen > 1024)
        {
            tinyfd_messageBox("错误", "数据库内用户名长度超限", "ok", "error", 1);
            return false;
        }
        PersonEntry entry;
        entry.name.resize(nameLen);
        ifs.read(entry.name.data(), nameLen);
        entry.features.resize(g_featureSize);
        ifs.read(reinterpret_cast<char*>(entry.features.data()),
            static_cast<std::streamsize>(g_featureSize * sizeof(float)));
        if (ifs.fail())
        {
            tinyfd_messageBox("错误", "读取特征库记录失败", "ok", "error", 1);
            return false;
        }
        g_database.push_back(std::move(entry));
    }
    return true;
}
// L2 归一化：将特征向量归一化到单位长度，使得余弦相似度计算更稳定
void l2Normalize(std::vector<float>& v) {
    double norm = 0.0;
    for (float val : v) norm += static_cast<double>(val) * val;
    norm = std::sqrt(norm);
    if (norm < 1e-12) return;  // 零向量不处理
    float invNorm = static_cast<float>(1.0 / norm);
    for (float& val : v) val *= invNorm;
}
// 余弦相似度
double cosineSimilarity(const float* a, const float* b, int dim) {
    double dot = 0.0, normA = 0.0, normB = 0.0;
    for (int i = 0; i < dim; ++i) {
        double va = static_cast<double>(a[i]);
        double vb = static_cast<double>(b[i]);
        dot += va * vb;
        normA += va * va;
        normB += vb * vb;
    }
    double denom = std::sqrt(normA) * std::sqrt(normB);
    if (denom < 1e-12) return 0.0;
    return dot / denom;
}
// CLAHE 预处理
cv::Mat preprocessImage(const cv::Mat& imgBGR) {
    cv::Mat lab, result;
    cv::cvtColor(imgBGR, lab, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> channels(3);
    cv::split(lab, channels);
    // 仅对 L 通道做 CLAHE（clipLimit=2.0, tileSize=8x8）
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(channels[0], channels[0]);
    cv::merge(channels, lab);
    cv::cvtColor(lab, result, cv::COLOR_Lab2BGR);
    return result;
}
// 查找目标图片
std::string findFaceImage() {
    for (const auto& entry : std::filesystem::directory_iterator(".")) {
        if (!entry.is_regular_file()) continue;
        std::string name = entry.path().filename().string();
        std::string nlower = name;
        std::transform(nlower.begin(), nlower.end(), nlower.begin(),
            [](unsigned char c) { return std::tolower(c); });
        if (nlower.size() > 5 && nlower.substr(0, 5) == "face.") {
            std::string ext = nlower.substr(4);
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" ||
                ext == ".bmp" || ext == ".tiff" || ext == ".tif" ||
                ext == ".webp") {
                return name;
            }
        }
    }
    return "";
}
// 特征提取：对单个人脸区域提取特征并做 L2 归一化
bool extractFeature(const SeetaImageData& img,
    const SeetaRect& rect,
    seeta::FaceLandmarker& landmarker,
    seeta::FaceRecognizer& recognizer,
    std::vector<float>& outFeature) {
    SeetaPointF points[5];
    landmarker.mark(img, rect, points);               // 5 点关键点定位
    outFeature.resize(g_featureSize);
    recognizer.Extract(img, points, outFeature.data()); // 提取特征
    l2Normalize(outFeature);                           // L2 归一化
    return true;
}
// 按人聚合匹配：将特征库中同一人的多条记录分组，取 top-3 平均相似度作为该人的得分；返回最佳匹配结果（人名、得分、第二名得分）
struct MatchResult {
    std::string name;
    double score;
    double secondScore;
};
MatchResult identifyPerson(const std::vector<float>& targetFeature) {
    // 按人名分组
    std::map<std::string, std::vector<const std::vector<float>*>> groups;
    for (const auto& entry : g_database) {
        groups[entry.name].push_back(&entry.features);
    }
    // 计算每个人与目标特征的相似度（取 top-3 平均）
    std::vector<std::pair<std::string, double>> personScores;
    personScores.reserve(groups.size());
    for (const auto& [name, feats] : groups) {
        std::vector<double> sims;
        sims.reserve(feats.size());
        for (const auto* f : feats) {
            sims.push_back(cosineSimilarity(targetFeature.data(), f->data(), g_featureSize));
        }
        // 降序排序，取前 3 个平均
        std::sort(sims.begin(), sims.end(), std::greater<double>());
        int topK = std::min(3, static_cast<int>(sims.size()));
        double avg = 0.0;
        for (int i = 0; i < topK; ++i) avg += sims[i];
        avg /= topK;
        personScores.emplace_back(name, avg);
    }
    // 按得分降序排列
    std::sort(personScores.begin(), personScores.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });
    MatchResult result;
    result.name = personScores.empty() ? "" : personScores[0].first;
    result.score = personScores.empty() ? 0.0 : personScores[0].second;
    result.secondScore = personScores.size() < 2 ? 0.0 : personScores[1].second;
    return result;
}
struct CaptureState {
    bool clicked = false;       // 是否已点击
    SeetaRect lastFaceRect;     // 当前帧最近人脸区域
    bool   hasFace = false;     // 当前帧是否检测到人脸
};
// 拍照（摄像头和窗口保持打开，只拍一张，返回保存路径）
std::string captureFace(cv::VideoCapture& cap,
    seeta::FaceDetector* detector,
    const std::string& winName,
    CaptureState& state) {
    state.clicked = false;
    state.hasFace = false;

    std::string savedPath;
    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        cv::Mat display = frame.clone();
        cv::flip(display, display, 1);

        cv::Mat proc = preprocessImage(display);
        SeetaImageData seetaImg;
        seetaImg.data = proc.data;
        seetaImg.width = proc.cols;
        seetaImg.height = proc.rows;
        seetaImg.channels = proc.channels();
        SeetaFaceInfoArray faces = detector->detect(seetaImg);

        int bestIdx = -1;
        int maxArea = 0;
        for (int i = 0; i < faces.size; ++i) {
            int area = faces.data[i].pos.width * faces.data[i].pos.height;
            if (area > maxArea) { maxArea = area; bestIdx = i; }
        }
        if (bestIdx >= 0) {
            const SeetaRect& r = faces.data[bestIdx].pos;
            state.lastFaceRect = r;
            state.hasFace = true;
            cv::rectangle(display, cv::Rect(r.x, r.y, r.width, r.height),
                cv::Scalar(0, 255, 0), 2);
        }
        else {
            state.hasFace = false;
            state.clicked = false;
        }
        cv::putText(display, "Click to capture | ESC to close",
            cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
            0.6, cv::Scalar(0, 255, 0), 2);
        cv::imshow(winName, display);

        int key = cv::waitKey(1) & 0xFF;
        if (key == 27) break;

        if (state.clicked && state.hasFace) {
            state.clicked = false;
            const SeetaRect& r = state.lastFaceRect;
            if (r.width >= 40 && r.height >= 40) {
                int marginX = static_cast<int>(r.width * 0.15);
                int marginY = static_cast<int>(r.height * 0.15);
                int cx = (std::max)(0, r.x - marginX);
                int cy = (std::max)(0, r.y - marginY);
                int cw = (std::min)(display.cols - cx, r.width + 2 * marginX);
                int ch = (std::min)(display.rows - cy, r.height + 2 * marginY);
                cv::Mat faceCrop = display(cv::Rect(cx, cy, cw, ch)).clone();
                savedPath = "face_capture.jpg";
                cv::imwrite(savedPath, faceCrop);
                std::cout << "Face captured -> " << savedPath
                    << " (" << cw << "x" << ch << ")" << std::endl;
                break;
            }
        }
    }
    return savedPath;
}
// 人脸识别
std::string facesearch(const std::string& detectorModel,
    const std::string& landmarkerModel,
    const std::string& recognizerModel,
    const std::string& dbPath,
    const std::string& imagePath = "") {
    g_detectorModel = detectorModel;
    g_landmarkerModel = landmarkerModel;
    g_recognizerModel = recognizerModel;
    g_dbPath = dbPath;
    // 加载三个模型
    seeta::FaceDetector* detector = nullptr;
    seeta::FaceLandmarker* landmarker = nullptr;
    seeta::FaceRecognizer* recognizer = nullptr;
    try {
        seeta::ModelSetting ds;
        ds.append(g_detectorModel.c_str());
        ds.set_device(seeta::ModelSetting::CPU);
        detector = new seeta::FaceDetector(ds);
        detector->set(seeta::FaceDetector::PROPERTY_MIN_FACE_SIZE, 20);
        detector->set(seeta::FaceDetector::PROPERTY_THRESHOLD, DETECT_THRESHOLD_HI);
        seeta::ModelSetting ls;
        ls.append(g_landmarkerModel.c_str());
        landmarker = new seeta::FaceLandmarker(ls);
        seeta::ModelSetting rs;
        rs.append(g_recognizerModel.c_str());
        rs.set_device(seeta::ModelSetting::CPU);
        recognizer = new seeta::FaceRecognizer(rs);
        // 从模型自动获取特征维度
        g_featureSize = recognizer->GetExtractFeatureSize();
        std::cout << "Models loaded. Feature dimension: " << g_featureSize << std::endl;
    }
    catch (const std::exception& e) {
        tinyfd_messageBox("模型加载失败", ("异常:" + std::string(e.what())).c_str(), "ok", "error", 1);
        delete detector; delete landmarker; delete recognizer;
        return "";
    }
    catch (...) {
        tinyfd_messageBox("模型加载失败", "未知异常", "ok", "error", 1);
        delete detector; delete landmarker; delete recognizer;
        return "";
    }
    // 加载特征库
    if (!loadDatabase(g_dbPath))
    {
        delete detector; delete landmarker; delete recognizer;
        return "";
    }
    std::cout << "Database loaded: " << g_database.size() << " records" << std::endl;
    if (g_database.empty())
    {
        tinyfd_messageBox("特征库为空", "特征库没有任何人脸数据", "ok", "error", 1);
        delete detector; delete landmarker; delete recognizer;
        return "";
    }
    // 确定目标图片路径
    std::string targetPath = imagePath.empty() ? findFaceImage() : imagePath;
    if (targetPath.empty())
    {
        tinyfd_messageBox("未找到人脸图片", "程序目录无face.xxx图片", "ok", "error", 1);
        delete detector; delete landmarker; delete recognizer;
        return "";
    }
    std::cout << "Target image: " << targetPath << std::endl;
    // 读取图片
    cv::Mat imgRaw = cv::imread(targetPath, cv::IMREAD_COLOR);
    if (imgRaw.empty())
    {
        tinyfd_messageBox("图片读取失败", ("无法打开文件:" + targetPath).c_str(), "ok", "error", 1);
        delete detector; delete landmarker; delete recognizer;
        return "";
    }
    // CLAHE 预处理，增强光照鲁棒性
    cv::Mat img = preprocessImage(imgRaw);
    // 人脸检测：先高阈值，失败则降级
    SeetaImageData seetaImg;
    seetaImg.data = img.data;
    seetaImg.width = img.cols;
    seetaImg.height = img.rows;
    seetaImg.channels = img.channels();
    SeetaFaceInfoArray faces = detector->detect(seetaImg);
    if (faces.size == 0) {
        // 降级阈值重试
        detector->set(seeta::FaceDetector::PROPERTY_THRESHOLD, DETECT_THRESHOLD_LO);
        faces = detector->detect(seetaImg);
        detector->set(seeta::FaceDetector::PROPERTY_THRESHOLD, DETECT_THRESHOLD_HI);
        if (faces.size == 0)
        {
            std::cout << "No face detected in the image" << std::endl;
            delete detector; delete landmarker; delete recognizer;
            return "";
        }
        std::cout << "Detected face(s) with fallback threshold" << std::endl;
    }
    std::cout << "Detected " << faces.size << " face(s)" << std::endl;
    // 对每张检测到的人脸提取特征，取最佳匹配
    MatchResult bestResult;
    bestResult.score = -1.0;
    for (int fi = 0; fi < faces.size; ++fi) {
        std::vector<float> feature;
        if (!extractFeature(seetaImg, faces.data[fi].pos, *landmarker, *recognizer, feature)) {
            continue;
        }
        MatchResult result = identifyPerson(feature);
        if (result.score > bestResult.score) {
            bestResult = result;
        }
    }
    // 输出比对详情
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Best match:  " << bestResult.name << " (" << bestResult.score << ")" << std::endl;
    if (!bestResult.name.empty()) {
        std::cout << "2nd match:   " << bestResult.secondScore << std::endl;
        std::cout << "Margin:      " << (bestResult.score - bestResult.secondScore) << std::endl;
    }
    std::cout << "Threshold:   " << RECOGNITION_THRESHOLD << std::endl;
    std::cout << "Margin min:  " << MARGIN_THRESHOLD << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    // 阈值 + 边距双重判定
    bool passScore = bestResult.score >= RECOGNITION_THRESHOLD;
    bool passMargin = (bestResult.score - bestResult.secondScore) >= MARGIN_THRESHOLD;
    std::string ret;
    if (passScore && passMargin) {
        // 得分达标且差距足够  确认识别
        std::cout << "Result: " << bestResult.name << std::endl;
        ret = bestResult.name;
    }
    else if (passScore && !passMargin) {
        // 得分达标但差距不够  可能混淆
        std::cout << "Result: UNCERTAIN — " << bestResult.name
            << " (margin too small, possible confusion)" << std::endl;
        ret = "UNCERTAIN";
    }
    else {
        // 得分不达标  未知
        std::cout << "Result: UNKNOWN" << std::endl;
        ret = "UNKNOWN";
    }
    std::cout << "----------------------------------------" << std::endl;
    delete detector;
    delete landmarker;
    delete recognizer;
    return ret;
}
// 超分处理
void facedetail() {
    // 读取输入图像
    cv::Mat src = cv::imread("face_capture.jpg", cv::IMREAD_COLOR);
    if (src.empty())
    {
        tinyfd_messageBox("超分图片读取失败", "找不到face_capture.jpg", "ok", "error", 1);
        return;
    }
    std::cout << "[Info] Input size: " << src.cols << "x" << src.rows << std::endl;
    // 自动查找超分模型文件
    const std::string modelName = "espcn";
    const int         scale = 2;
    const std::vector<std::string> searchPaths = {
        "models/ESPCN_x2.pb",
    };
    std::string modelFile;
    for (const auto& p : searchPaths) {
        std::ifstream ifs(p, std::ios::binary);
        if (ifs.good()) { modelFile = p; break; }
    }
    if (modelFile.empty())
    {
        tinyfd_messageBox("超分模型缺失", "找不到models/ESPCN_x2.pb", "ok", "error", 1);
        return;
    }
    std::cout << "[Info] Set models: " << modelFile << std::endl;
    // 加载模型并执行超分
    cv::dnn_superres::DnnSuperResImpl sr;
    sr.readModel(modelFile);
    sr.setModel(modelName, scale);
    cv::Mat result;
    sr.upsample(src, result);
    // 保存结果
    cv::imwrite("face_capture.jpg", result);
    std::cout << "[Successful] Saved face_capture.jpg ("
        << result.cols << "x" << result.rows << ")" << std::endl;
}

int main(int argc, char* argv[]) {
    // 初始化
    remove("face_capture.jpg");
    std::string configPath = "config.json";
    std::ifstream fin(configPath);

    if (!fin.is_open())
    {
        const char* filter[] = { "*.json" };
        const char* selected = tinyfd_openFileDialog("选择配置文件", "", 1, filter, "JSON配置文件", 0);
        if (!selected)
        {
            tinyfd_messageBox("错误", "未选择配置文件，程序退出", "ok", "error", 1);
            return 1;
        }
        configPath = selected;
        fin.open(configPath);
        if (!fin.is_open())
        {
            tinyfd_messageBox("错误", "无法打开配置文件", "ok", "error", 1);
            return 1;
        }
    }

    nlohmann::json j;
    try
    {
        fin >> j;
    }
    catch (const nlohmann::json::parse_error& e)
    {
        tinyfd_messageBox("JSON解析失败", e.what(), "ok", "error", 1);
        return 1;
    }

    // 读取配置
    cameraId = j.value("camera_id", 0);
    cap_width_set = j.value("cap_width", 1200);
    cap_height_set = j.value("cap_height", 800);
    RECOGNITION_THRESHOLD = j.value("RECOGNITION_THRESHOLD", 0.7f);
    MARGIN_THRESHOLD = j.value("MARGIN_THRESHOLD", 0.15f);
    DETECT_THRESHOLD_HI = j.value("DETECT_THRESHOLD_HI", 0.9f);
    DETECT_THRESHOLD_LO = j.value("DETECT_THRESHOLD_LO", 0.7f);

    fin.close();

    std::string detectorModel = "models/face_detector.csta";
    std::string landmarkerModel = "models/face_landmarker_pts5.csta";
    std::string recognizerModel = "models/face_recognizer.csta";
    std::string dbPath = "face_features.db";

    if (argc >= 5) {
        detectorModel = argv[1];
        landmarkerModel = argv[2];
        recognizerModel = argv[3];
        dbPath = argv[4];
    }
    else if (argc >= 4) {
        detectorModel = argv[1];
        landmarkerModel = argv[2];
        recognizerModel = argv[3];
    }
    if (argc >= 6) {
        cameraId = std::stoi(argv[5]);
    }

    // 一：打开摄像头（仅一次）
    remove("face_capture.jpg");

    // 初始化检测器
    seeta::FaceDetector* detector = nullptr;
    try {
        seeta::ModelSetting ds;
        ds.append(detectorModel.c_str());
        ds.set_device(seeta::ModelSetting::CPU);
        detector = new seeta::FaceDetector(ds);
        detector->set(seeta::FaceDetector::PROPERTY_MIN_FACE_SIZE, 20);
        detector->set(seeta::FaceDetector::PROPERTY_THRESHOLD, DETECT_THRESHOLD_HI);
    }
    catch (const std::exception& e) {
        tinyfd_messageBox("检测器初始化错误", ("初始化失败:" + std::string(e.what())).c_str(), "ok", "error", 1);
        std::this_thread::sleep_for(std::chrono::seconds(1));
        return 1;
    }
    catch (...) {
        tinyfd_messageBox("检测器初始化错误", "未知异常", "ok", "error", 1);
        std::this_thread::sleep_for(std::chrono::seconds(1));
        return 1;
    }

    // 打开摄像头
    cv::VideoCapture cap;
    if (!cap.open(cameraId))
    {
        cap.open(cameraId, cv::CAP_DSHOW);
        if (!cap.isOpened())
        {
            tinyfd_messageBox("摄像头打开失败", ("无法打开摄像头ID:" + std::to_string(cameraId)).c_str(), "ok", "error", 1);
            std::this_thread::sleep_for(std::chrono::seconds(1));
            delete detector;
            return 1;
        }
    }
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, cap_width_set);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, cap_height_set);
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

    // 创建持久窗口
    const std::string winName = "Face Capture";
    cv::namedWindow(winName, cv::WINDOW_NORMAL);
    cv::resizeWindow(winName, 640, 480);
    CaptureState captureState;
    cv::setMouseCallback(winName,
        [](int event, int, int, int, void* userdata) {
            if (event == cv::EVENT_LBUTTONDOWN)
                static_cast<CaptureState*>(userdata)->clicked = true;
        }, &captureState);

    // 主循环：拍照 确认 超分 识别 确认身份 不确认就重拍
    std::string capturedPath;
    std::string result;
    while (true)
    {
        // 1. 拍照
        capturedPath = captureFace(cap, detector, winName, captureState);
        if (capturedPath.empty())
        {
            tinyfd_messageBox("取消", "拍照已取消", "ok", "info", 1);
            cv::destroyWindow(winName);
            delete detector;
            return 0;
        }

        // 2. 超分处理
        facedetail();

        // 3. 人脸识别
        result = facesearch(detectorModel, landmarkerModel, recognizerModel, dbPath, capturedPath);
        std::cout << "Recognition result: " << result << std::endl;

        if (result == "UNKNOWN" || result == "" || result == "UNCERTAIN")
        {
            tinyfd_messageBox("人脸识别失败", ("识别结果: " + result + "\n\n请重新拍照").c_str(), "ok", "error", 1);
            continue;  // 识别失败 重拍，摄像头不关
        }

        // 4. 确认身份
        std::string tip = "识别结果：\n\n" + result + "\n\n是否确认身份？\n点「否」重新拍照";
        int confirm = tinyfd_messageBox("确认身份", tip.c_str(), "yesno", "question", 1);
        if (confirm == 1) break;  // 确认 退出循环，上传数据
    }

    // 身份确认完毕，关闭摄像头和窗口
    cv::destroyWindow(winName);
    delete detector;
    return 0;
}