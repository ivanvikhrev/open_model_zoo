/*
// Copyright (C) 2018-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#pragma once
#include <samples/ocv_common.hpp>

struct InputData {
    virtual ~InputData() {}

    template<class T> T& asRef() {
        return dynamic_cast<T&>(*this);
    }

    template<class T> const T& asRef() const {
        return dynamic_cast<const T&>(*this);
    }
};

struct ImageInputData : public InputData{
    cv::Mat inputImage;

    ImageInputData() {}
    ImageInputData(const cv::Mat& img) {
        inputImage = img;
    }
};

struct ImageRetinaFaceInputData : public ImageInputData {
    std::vector<std::vector<cv::Point2f>> landmarks_regression;
    std::vector<cv::Mat> masks;

    ImageRetinaFaceInputData() : ImageInputData() {}
    ImageRetinaFaceInputData(const cv::Mat& img) : ImageInputData(img) {}
    ImageRetinaFaceInputData(std::vector<std::vector<cv::Point2f>> landmarks_regression) : landmarks_regression(landmarks_regression) {}

};
