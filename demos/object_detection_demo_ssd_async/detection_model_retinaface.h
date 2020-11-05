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
#include <vector>
#include <ngraph/ngraph.hpp>

class ModelRetinaFace :
    public DetectionModel
{
protected:
    struct AnchorCfgLine
    {
        int stride;
        std::vector<double> scales;
        int baseSize;
        std::vector<double> ratios;
    };

public:
    struct Anchor
    {
        double left;
        double top;
        double right;
        double bottom;

        double getWidth() const { return (right - left) +1; }
        double getHeight() const { return (bottom - top) + 1; }
        double getXCenter() const { return left + (getWidth() - 1.0) / 2.; }
        double getYCenter() const { return top + (getHeight() - 1.0) / 2.; }
    };

public:
    static const int LANDMARKS_NUM = 5;

    /// Loads model and performs required initialization
    /// @param model_name name of model to load
    /// @param cnnConfig - fine tuning configuration for CNN model
    /// @param confidenceThreshold - threshold to eleminate low-confidence detections.
    /// Any detected object with confidence lower than this threshold will be ignored.
    /// @param useAutoResize - if true, image will be resized by IE.
    /// @param shouldDetectMasks - if true, masks will be detected.
    /// Otherwise, image will be preprocessed and resized using OpenCV routines.
    /// @param labels - array of labels for every class. If this array is empty or contains less elements
    /// than actual classes number, default "Label #N" will be shown for missing items.
    /// @param engine - pointer to InferenceEngine::Core instance to use.
    /// If it is omitted, new instance of InferenceEngine::Core will be created inside.
    ModelRetinaFace(const std::string& model_name, float confidenceThreshold, bool useAutoResize,
        bool shouldDetectMasks = false, const std::vector<std::string>& labels = std::vector<std::string>());
  /*  virtual void init(const std::string& model_name, const CnnConfig& cnnConfig,
        float confidenceThreshold, bool useAutoResize, bool shouldDetectMasks=false,
        const std::vector<std::string>& labels = std::vector<std::string>(),
        InferenceEngine::Core* engine = nullptr);*/

    // virtual void onLoadCompleted(InferenceEngine::ExecutableNetwork* execNetwork, RequestsPool* requestsPool);
    std::unique_ptr<ResultBase> postprocess(InferenceResult & infResult);
protected:
    virtual void prepareInputsOutputs(InferenceEngine::CNNNetwork & cnnNetwork);
    void generate_anchors_fpn();
    //std::vector<DetectionPipeline::ObjectDesc> process_output(PipelineBase::InferenceResult infResult, double scale_x, double scale_y, double face_prob_threshold);

    bool shouldDetectMasks = false;
    std::vector <AnchorCfgLine> anchorCfg;
    std::map<int, std::vector <Anchor>> _anchors_fpn;
    double landmark_std;

    enum EOutputType {
        OT_BBOX,
        OT_SCORES,
        OT_LANDMARK,
        OT_MASKSCORES,
        OT_MAX
    };
    std::vector <std::string> separateOutputsNames[OT_MAX];
};

