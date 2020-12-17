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
#include "detection_model.h"
class ModelFaceBoxes : public DetectionModel {
public:
    struct Anchor {
        double cx;
        double cy;
        double skx;
        double sky;

        //double getWidth() const { return (right - left) + 1; }
        //double getHeight() const { return (bottom - top) + 1; }
        //double getXCenter() const { return left + (getWidth() - 1.0) / 2.; }
        //double getYCenter() const { return top + (getHeight() - 1.0) / 2.; }
    };
    std::vector<std::vector<int>> minSizes;
    ModelFaceBoxes(const std::string& modelFileName, float confidenceThreshold, bool useAutoResize, float boxIOUThreshold);
    std::unique_ptr<ResultBase> postprocess(InferenceResult& infResult) override;

protected:
    std::vector<Anchor> priorBoxes(std::vector<std::pair<int, int>> featureMaps, int imgWidth, int imgHeight);
    virtual void prepareInputsOutputs(InferenceEngine::CNNNetwork& cnnNetwork) override;
    double boxIOUThreshold;
    int keepTopK;
    std::array<double, 2> variance;
    std::array<int, 3> steps;
};
