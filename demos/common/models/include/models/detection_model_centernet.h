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

class ModelCenterNet : public DetectionModel {
public:
    struct BBoxes {
        float left;
        float top;
        float right;
        float bottom;

        float getWidth() const { return (right - left) + 1; }
        float getHeight() const { return (bottom - top) + 1; }
    };
    static const int INIT_VECTOR_SIZE = 200;

    ModelCenterNet(const std::string& modelFileName, float confidenceThreshold, bool useAutoResize,
        const std::vector<std::string>& labels = std::vector<std::string>());
    std::shared_ptr<InternalModelData> preprocess(
        const InputData& inputData, InferenceEngine::InferRequest::Ptr& request) override;
    std::unique_ptr<ResultBase> postprocess(InferenceResult& infResult) override;

protected:
    virtual void prepareInputsOutputs(InferenceEngine::CNNNetwork& cnnNetwork) override;
    std::string getLabelName(int labelID) { return (size_t)labelID < labels.size() ? labels[labelID] : std::string("Label #") + std::to_string(labelID); }

};