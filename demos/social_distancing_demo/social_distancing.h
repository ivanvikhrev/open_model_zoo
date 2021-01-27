#pragma once

#include <models/results.h>
#include <tuple>
//#include "opencv2/opencv.hpp"

// #include "common/no_copy.h"
// #include "common/types.h"
// #include "common/frame.h"
// #include "common/logic.h"
// #include "common/presenter.h"
// #include "detector/yolo_v2_detector.h"

class SocialDistancing {
public:
    // using Ptr = std::shared_ptr<SocialDistancing>;
    // using SDOut = std::tuple<std::vector<cv::Rect>, std::vector<std::tuple<int32_t, int32_t, float>>>;
    // using SDMeta = Meta<SDOut>;


public:
    SocialDistancing();

    // ILogic interface
public:
    virtual void create(const std::string &json_obj_str);
    virtual std::unique_ptr<ResultBase> process(const DetectionResult& result, size_t personID);
    //virtual SDOut process(const std::vector<pz::DetectedObject> &objects);

private:
    float cal_dist_;
    std::vector<cv::Point2f> cal_pts_;
    cv::Mat homography_;
};



// class SocialDistancingPresenter: public IPresenter
// {
// public:
//     static int32_t sId;
//     using Ptr = std::shared_ptr<SocialDistancingPresenter>;

// public:
//     SocialDistancingPresenter(bool show_image = true, double scale = 1.0);
//     virtual ~SocialDistancingPresenter();


// public:
//     virtual void setParameters(const std::string &config_fn);
//     virtual void present(const Frame::Ptr &frame, const MetaBase::Ptr &meta);

// private:
//     int32_t id_;
//     std::string win_name_;
//     bool show_image_;
//     double scale_;
// };


// }




