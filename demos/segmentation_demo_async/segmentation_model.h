/*
// Copyright (C) 2018-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writingb  software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include "model_base.h"
#include "opencv2/core.hpp"

#pragma once
class SegmentationModel :
    public ModelBase
{
public:
    /// Constructor
    /// @param model_nameFileName of model to load
    SegmentationModel(const std::string& modelFileName);

    virtual void preprocess(const InputData& inputData, InferenceEngine::InferRequest::Ptr& request, MetaData*& metaData);
    virtual std::unique_ptr<ResultBase> postprocess(InferenceResult& infResult);
    virtual cv::Mat renderData(ResultBase* result);

protected:
    virtual void prepareInputsOutputs(InferenceEngine::CNNNetwork & cnnNetwork);
    const cv::Vec3b& class2Color(int classId);

    int outHeight = 0;
    int outWidth = 0;
    int outChannels = 0;

    std::vector<cv::Vec3b> colors;
    std::mt19937 rng;
    std::uniform_int_distribution<int> distr;
};
