#include <algorithm>

#include "social_distancing.h"
// #include "rapidjson/rapidjson.h"
// #include "rapidjson/document.h"
// #include "rapidjson/filereadstream.h"

// using ::rapidjson::Document;
// using ::rapidjson::FileReadStream;

const static std::vector<cv::Point2f> kWorldPoints = {
        cv::Point2f(-0.5f * 1.f, -0.5f * 1.f),
        cv::Point2f( 0.5f * 1.f, -0.5f * 1.f),
        cv::Point2f( 0.5f * 1.f,  0.5f * 1.f),
        cv::Point2f(-0.5f * 1.f,  0.5f * 1.f)
    };

SocialDistancing::SocialDistancing() {
    cal_dist_ = 2.0;
    cal_pts_.push_back({ 690, 522 });
    cal_pts_.push_back({ 859, 513 });
    cal_pts_.push_back({ 900, 634 });
    cal_pts_.push_back({ 709, 645 });
    homography_ = cv::findHomography(cal_pts_, kWorldPoints);
}

void SocialDistancing::create(const std::string &json_fn)
{
    // FILE* fp = fopen(json_fn.c_str(), "rb");

    // if (NULL == fp)
    //     throw std::runtime_error("Cannot open " + json_fn);


    // char buffer[65536];
    // FileReadStream is(fp, buffer, sizeof(buffer));

    // fclose(fp);

    // Document d;
    // d.ParseStream(is);

    ///
    /// cal_distance
    ///
    // if (d.HasMember("cal_distance") && d["cal_distance"].IsFloat())
    // {
    //     cal_dist_ = d["cal_distance"].GetFloat();
    // }
    // else
    // {
    //     throw std::runtime_error("Json must have 'cal_distance' property.");
    // }

    ///
    /// cal_pts
    ///
    //if (d.HasMember("cal_pts") && d["cal_pts"].IsArray())
    // {
    //     for (auto& v : d["cal_pts"].GetArray())
    //     {
    //         std::vector<float> vals;
    //         for (auto& a : v.GetArray())
    //             vals.push_back(a.GetFloat());

    //         cv::Point2f p(vals.at(0), vals.at(1));
    //         cal_pts_.push_back(p);
    //     }
    //     if (cal_pts_.size() != 4)
    //         throw std::runtime_error("Number of calibration points should be 4.");
    // }
    // else
    // {
    //     throw std::runtime_error("Json must have calibration points.");
    // }
    cal_dist_ = 2.0;
    cal_pts_.push_back({690, 522});
    cal_pts_.push_back({859, 513});
    cal_pts_.push_back({900, 634});
    cal_pts_.push_back({709, 645});
    //homography_ = cv::findHomography(cal_pts_, kWorldPoints);
}

std::unique_ptr<ResultBase> SocialDistancing::process(const DetectionResult& detRes, size_t personID) {
    if (!detRes.metaData) {
        throw std::invalid_argument("Renderer: metadata is null");
    }

    auto outputImg = detRes.metaData->asRef<ImageMetaData>().img;
    auto& detected = detRes.objects;

    std::vector<DetectedObject> people;
    std::copy_if(detected.begin(), detected.end(), std::back_inserter(people),
                 [personID](const DetectedObject& o) { return o.labelID == personID; });

    std::vector<cv::Rect> rects;
    std::transform(std::begin(people), std::end(people),
                   std::back_inserter(rects),
                [](const DetectedObject& o) {
                    return cv::Rect(o.x, o.y, o.width, o.height);
                });

    std::vector<cv::Point2f> origins;
    std::transform(std::begin(rects), std::end(rects),
                   std::back_inserter(origins),
                [](cv::Rect& r) {
                    return cv::Point2f(r.x + (r.width * 0.5f), r.height);
                });

    if (origins.size() == 0)
        return nullptr;

    std::vector<cv::Point2f> transformed(origins.size());
    cv::perspectiveTransform(origins, transformed, homography_);

    std::vector<std::tuple<int32_t, int32_t, float>> edges;
    for (int i = 0 ; i < origins.size(); ++i) {
        for (int j = i + 1 ; j < origins.size(); ++j) {
            auto p0 = origins[i];
            auto p1 = origins[j];
            auto tp0 = transformed[i];
            auto tp1 = transformed[j];

            // calculates distance between tp0 and tp1.
            auto v = tp0 - tp1;
            float dist = std::sqrt(v.x * v.x + v.y * v.y) * cal_dist_;
            edges.push_back(std::make_tuple(i, j, dist));
        }
    }
    SocialDistanceResult* result = new SocialDistanceResult;
    *static_cast<ResultBase*>(result) = static_cast<const ResultBase&>(detRes);

    // auto meta = std::make_shared<SDMeta>();
    // meta->data() = std::make_tuple(rects, edges);
    //for (auto& r : rects) {
    //    DetectedObject desc;
    //    desc.confidence = scores[i].second;
    //    desc.labelID = scores[i].first / chSize;
    //    desc.label = getLabelName(desc.labelID);
    //    desc.x = bboxes[i].left;
    //    desc.y = bboxes[i].top;
    //    desc.width = bboxes[i].getWidth();
    //    desc.height = bboxes[i].getHeight();

    //    result->objects.push_back(desc);
    //}
    result->objects = people;
    result->edges = edges;
    return std::unique_ptr<ResultBase>(result);;
}

// SocialDistancing::SDOut SocialDistancing::process(const std::vector<pz::DetectedObject> &objects)
// {
//     SocialDistancing::SDOut out;

//     if (objects.size() == 0)
//         return out;

//     std::vector<pz::DetectedObject> people;
//     std::copy_if(objects.cbegin(), objects.cend(), std::back_inserter(people),
//                  [](const pz::DetectedObject& o) { return o.labelId == 14; });

//     std::vector<cv::Rect> rects;
//     std::transform(std::begin(people), std::end(people),
//                    std::back_inserter(rects),
//                 [](pz::DetectedObject& o) {
//                     return cv::Rect(o.x, o.y, o.width, o.height);
//                 });

//     std::vector<cv::Point2f> origins;
//     std::transform(std::begin(rects), std::end(rects),
//                    std::back_inserter(origins),
//                 [](cv::Rect& r) {
//                     return cv::Point2f(r.x + (r.width * 0.5f), r.height);
//                 });

//     if (origins.size() == 0)
//         return out;

//     std::vector<cv::Point2f> transformed(origins.size());
//     cv::perspectiveTransform(origins, transformed, homography_);

//     std::vector<std::tuple<int32_t, int32_t, float>> edges;
//     for (int i = 0 ; i < origins.size() ; i++)
//     {
//         for (int j = i + 1 ; j < origins.size(); j++)
//         {
//             auto p0 = origins[i];
//             auto p1 = origins[j];
//             auto tp0 = transformed[i];
//             auto tp1 = transformed[j];

//             /// calculates distance between tp0 and tp1.
//             auto v = tp0 - tp1;
//             float dist = std::sqrt(v.x * v.x + v.y * v.y) * cal_dist_;
//             edges.push_back(std::make_tuple(i, j, dist));
//         }
//     }

//     return std::make_tuple(rects, edges);
// }

//int32_t SocialDistancingPresenter::sId = 0;
//
//SocialDistancingPresenter::SocialDistancingPresenter(bool show_image, double scale)
//    :id_(sId++), show_image_(show_image)
//{
//    std::stringstream ss;
//    ss << "preview [" << id_ << "]";
//    win_name_ = ss.str();
//    scale_ = scale;
//}
//
//SocialDistancingPresenter::~SocialDistancingPresenter()
//{
//    if (show_image_)
//        cv::destroyWindow(win_name_);
//}
//
//void SocialDistancingPresenter::setParameters(const std::string &config_fn)
//{
//
//}

//void SocialDistancingPresenter::present(const Frame::Ptr &frame, const MetaBase::Ptr &meta)
//{
//    if (frame == nullptr)
//        return;
//
//    if (show_image_)
//    {
//        cv::Mat mat;
//
//        if (frame->pixelFormat() == PixelFormat::NV12)
//        {
//            cv::cvtColor(frame->data(0), mat, cv::COLOR_YUV2BGR_NV12);
//        }
//        else
//        {
//            mat = frame->data(0);
//        }
//
//        if (meta != nullptr)
//        {
//            auto sdmeta = std::dynamic_pointer_cast<pz::SocialDistancing::SDMeta>(meta);
//
//            if (sdmeta != nullptr)
//            {
//                std::vector<cv::Rect> rois;
//                std::vector<std::tuple<int32_t, int32_t, float>> edges;
//                std::tie(rois, edges) = sdmeta->data();
//
//                for (auto& r : rois)
//                {
//                    cv::rectangle(mat, r, {0, 255, 0}, 2);
//                }
//
//                for(auto& e : edges)
//                {
//                    int32_t i;
//                    int32_t j;
//                    float dist;
//                    std::tie(i, j, dist) = e;
//                    auto& r0 = rois[i];
//                    auto& r1 = rois[j];
//                    auto p0 = cv::Point(r0.x + (r0.width/2), r0.y + r0.height);
//                    auto p1 = cv::Point(r1.x + (r1.width/2), r1.y + r1.height);
//
//                    std::stringstream ss;
//                    ss << std::fixed << std::setprecision(1)
//                        << round(dist * 10.0f) / 10.0f << "m";
//
//                    auto c = ((p0 + p1) * 0.5f);
//
//                    if (dist <= 2.0f)
//                    {
//                        cv::line(mat, p0, p1, cv::Scalar(0, 0, 255), 1);
//                        cv::putText(mat, ss.str(), c, cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(255, 255, 255), 4);
//                        cv::putText(mat, ss.str(), c, cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(0, 0, 255), 2);
//                    }
//                    else
//                    {
//                        cv::line(mat, p0, p1, cv::Scalar(255, 0, 0), 1);
//                        cv::putText(mat, ss.str(), c, cv::FONT_HERSHEY_PLAIN, 1.3, cv::Scalar(255, 255, 255), 4);
//                        cv::putText(mat, ss.str(), c, cv::FONT_HERSHEY_PLAIN, 1.3, cv::Scalar(255, 0, 0), 2);
//                    }
//                }
//            }
//        }
//        cv::Mat out_mat;
//        cv::resize(mat, out_mat, cv::Size(0, 0), scale_, scale_);
//        cv::imshow(win_name_, out_mat);
//        cv::waitKey(1);
//    }
//    else
//    {
//        auto sdmeta = std::dynamic_pointer_cast<pz::SocialDistancing::SDMeta>(meta);

//        if (sdmeta != nullptr)
//        {
//            std::vector<cv::Rect> rois;
//            std::vector<std::tuple<int32_t, int32_t, float>> edges;
//            std::tie(rois, edges) = sdmeta->data();
//            std::cout << "[" << std::setw(6) << frame->frameId() << "] "
//                    << "num rois: " << rois.size() << ", num edges: " << edges.size() << std::endl;
//        }
//        else
//        {
//            std::cout << "[" << std::setw(6) << frame->frameId() << "] num rois: 0, num edges: 0" << std::endl;
//        }
//    }
//}
//}
