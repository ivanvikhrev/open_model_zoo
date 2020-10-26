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

#include "detection_pipeline_ssd.h"
#include <samples/slog.hpp>
#include "detection_pipeline_retinaface.h"

using namespace InferenceEngine;
void DetectionPipelineRetinaface::init(const std::string& model_name, const CnnConfig& cnnConfig,
    float confidenceThreshold, bool useAutoResize, bool shouldDetectMasks,
    const std::vector<std::string>& labels,
    InferenceEngine::Core* engine) {

    DetectionPipeline::init(model_name, cnnConfig, confidenceThreshold, useAutoResize, labels, engine);

    this->shouldDetectMasks = shouldDetectMasks;
    anchorCfg.push_back({ 32, { 32,16 }, 16, { 1.0 } });
    anchorCfg.push_back({ 16, { 8,4 }, 16, { 1.0 } });
    anchorCfg.push_back({ 8, { 2,1 }, 16, { 1.0 } });

    generate_anchors_fpn();

    landmark_std = shouldDetectMasks ? 0.2 : 1.0;

}

DetectionPipeline::DetectionResult DetectionPipelineRetinaface::getProcessedResult(bool shouldKeepOrder)
{
    auto infResult = PipelineBase::getInferenceResult(shouldKeepOrder);
    if (infResult.IsEmpty()) {
        return DetectionResult();
    }

    DetectionResult result;

    static_cast<ResultBase&>(result) = static_cast<ResultBase&>(infResult);
    result.objects = process_output(infResult,((double)netInputWidth)/infResult.extraData.cols, ((double)netInputHeight) / infResult.extraData.rows,confidenceThreshold);

    return result;
}

void DetectionPipelineRetinaface::prepareInputsOutputs(InferenceEngine::CNNNetwork & cnnNetwork){
    // --------------------------- Configure input & output ---------------------------------------------
    // --------------------------- Prepare input blobs -----------------------------------------------------
    slog::info << "Checking that the inputs are as the demo expects" << slog::endl;
    InputsDataMap inputInfo(cnnNetwork.getInputsInfo());
    if (inputInfo.size() != 1) {
        throw std::logic_error("This demo accepts networks that have only one input");
    }
    InputInfo::Ptr& input = inputInfo.begin()->second;
    imageInputName = inputInfo.begin()->first;
    input->setPrecision(Precision::U8);
    if (useAutoResize) {
        input->getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
        input->getInputData()->setLayout(Layout::NHWC);
    }
    else {
        input->getInputData()->setLayout(Layout::NCHW);
    }

    //--- Reading image input parameters
    imageInputName = inputInfo.begin()->first;
    const TensorDesc& inputDesc = inputInfo.begin()->second->getTensorDesc();
    netInputHeight = getTensorHeight(inputDesc);
    netInputWidth = getTensorWidth(inputDesc);

    // --------------------------- Prepare output blobs -----------------------------------------------------
    slog::info << "Checking that the outputs are as the demo expects" << slog::endl;

    InferenceEngine::OutputsDataMap outputInfo(cnnNetwork.getOutputsInfo());

    std::vector<int> outputsSizes[OT_MAX];

    for (auto& output : outputInfo) {
        output.second->setPrecision(InferenceEngine::Precision::FP32);
        output.second->setLayout(InferenceEngine::Layout::NCHW);
        outputsNames.push_back(output.first);

        EOutputType type= OT_MAX;
        if (output.first.find("bbox") != -1) {
            type = OT_BBOX;
        }
        else if (output.first.find("cls") != -1) {
            type = OT_SCORES;
        }
        else if(output.first.find("landmark") != -1) {
            type = OT_LANDMARK;
        }
        else if(shouldDetectMasks && output.first.find("type") != -1) {
            type = OT_MASKSCORES;
        }
        else {
            continue;
        }

        size_t num = output.second->getDims()[2];
        size_t i = 0;
        for (; i < outputsSizes[type].size(); i++)
        {
            if (num < outputsSizes[type][i])
            {
                break;
            }
        }
        separateOutputsNames[type].insert(separateOutputsNames[type].begin()+i,output.first);
        outputsSizes[type].insert(outputsSizes[type].begin() + i, num);
    }

    if (outputsNames.size()!=9 && outputsNames.size() != 12)
        throw std::logic_error("Expected 12 or 9 output blobs");
}

std::vector<DetectionPipelineRetinaface::Anchor> _ratio_enum(const DetectionPipelineRetinaface::Anchor& anchor, std::vector<double> ratios) {
    std::vector<DetectionPipelineRetinaface::Anchor> retVal;
    auto w = anchor.getWidth();
    auto h = anchor.getHeight();
    auto xCtr = anchor.getXCenter();
    auto yCtr = anchor.getYCenter();
    for (auto ratio : ratios)
    {
        auto size = w * h;
        auto size_ratio = size / ratio;
        auto ws = std::round(sqrt(size_ratio));
        auto hs = std::round(ws * ratio);
        retVal.push_back({ xCtr - 0.5 * (ws - 1), yCtr - 0.5 * (hs - 1), xCtr + 0.5 * (ws - 1), yCtr + 0.5 * (hs - 1) });
    }
    return retVal;
}

std::vector<DetectionPipelineRetinaface::Anchor> _scale_enum(const DetectionPipelineRetinaface::Anchor& anchor, std::vector<double> scales) {
    std::vector<DetectionPipelineRetinaface::Anchor> retVal;
    auto w = anchor.getWidth();
    auto h = anchor.getHeight();
    auto xCtr = anchor.getXCenter();
    auto yCtr = anchor.getYCenter();
    for (auto scale : scales)
    {
        auto ws = w * scale;
        auto hs = h * scale;
        retVal.push_back({ xCtr - 0.5 * (ws - 1), yCtr - 0.5 * (hs - 1), xCtr + 0.5 * (ws - 1), yCtr + 0.5 * (hs - 1) });
    }
    return retVal;
}

    
std::vector<DetectionPipelineRetinaface::Anchor> generate_anchors(int base_size, const std::vector<double>& ratios, const std::vector<double>& scales) {
    DetectionPipelineRetinaface::Anchor base_anchor{ 0, 0, (double)base_size - 1, (double)base_size - 1 };
    auto ratio_anchors = _ratio_enum(base_anchor, ratios);
    std::vector<DetectionPipelineRetinaface::Anchor> retVal;

    for (auto ra : ratio_anchors)
    {
        auto addon = _scale_enum(ra, scales);
        retVal.insert(retVal.end(), addon.begin(), addon.end());
    }
    return retVal;
}

void DetectionPipelineRetinaface::generate_anchors_fpn()
{
    auto cfg = anchorCfg;
    std::sort(cfg.begin(), cfg.end(), [](AnchorCfgLine& x, AnchorCfgLine& y) {return x.stride > y.stride; });

    std::vector<DetectionPipelineRetinaface::Anchor> anchors;
    for (auto cfgLine : cfg)
    {
        auto anchors = generate_anchors(cfgLine.baseSize, cfgLine.ratios, cfgLine.scales);
        _anchors_fpn.emplace(cfgLine.stride,anchors);
    }
}

std::vector<int> nms(std::vector<DetectionPipelineRetinaface::Anchor> boxes, std::vector<double> scores, double thresh){
    std::vector<double> areas;
    for (int i = 0; i < boxes.size(); i++)
    {
        areas.push_back((boxes[i].right - boxes[i].left) * (boxes[i].bottom - boxes[i].top));
    }
    std::vector<int> order(scores.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&scores](int o1, int o2) { return scores[o1] > scores[o2]; });

    int ordersNum = 0;
    for (; scores[order[ordersNum]] >= 0; ordersNum++);

    std::vector<int> keep;
    bool shouldContinue = true;
    for(int i=0;shouldContinue && i<ordersNum;i++)
    {
        auto idx1 = order[i];
        if (idx1 >= 0)
        {
            keep.push_back(idx1);
            shouldContinue = false;

            for (int j = 1; j < ordersNum; j++)
            {
                auto idx2 = order[j];
                if (idx2 >= 0)
                {
                    shouldContinue = true;

                    double overlappingWidth = fmin(boxes[idx1].right, boxes[idx2].right) - fmax(boxes[idx1].left, boxes[idx2].left);
                    double overlappingHeight = fmin(boxes[idx1].bottom, boxes[idx2].bottom) - fmax(boxes[idx1].top, boxes[idx2].top);

                    auto intersection = overlappingWidth > 0 && overlappingHeight > 0 ? overlappingWidth * overlappingHeight : 0;

                    auto overlap = intersection / (areas[idx1] + areas[idx2] - intersection);
                    if (overlap >= thresh)
                    {
                        order[j]=-1;
                    }
                }
            }
        }
    }
    return keep;
}

std::vector<DetectionPipelineRetinaface::Anchor> _get_proposals(InferenceEngine::MemoryBlob::Ptr rawData, int anchor_num, const std::vector<DetectionPipelineRetinaface::Anchor>& anchors) {
    auto desc = rawData->getTensorDesc();
    auto sz = desc.getDims();

    std::vector<DetectionPipelineRetinaface::Anchor> retVal;

    LockedMemory<const void> outputMapped = rawData->rmap();
    const float *memPtr = outputMapped.as<float*>();
    auto bbox_pred_len = sz[1] / anchor_num;
    auto blockWidth = sz[2] * sz[3];

    for (int i = 0; i < anchors.size(); i++) {
        auto offset = blockWidth * bbox_pred_len * (i % anchor_num) + (i / anchor_num);
        auto dx = memPtr[offset];
        auto dy = memPtr[offset + bbox_pred_len];
        auto dw = memPtr[offset + bbox_pred_len * 2];
        auto dh = memPtr[offset + bbox_pred_len * 3];

        auto pred_ctr_x = dx * anchors[i].getWidth() + anchors[i].getXCenter();
        auto pred_ctr_y = dx * anchors[i].getHeight() + anchors[i].getYCenter();
        auto pred_w = exp(dw) * anchors[i].getWidth();
        auto pred_h = exp(dh) * anchors[i].getHeight();
        retVal.push_back({ pred_ctr_x - 0.5 * (pred_w - 1.0), pred_ctr_y - 0.5 * (pred_h - 1.0),
            pred_ctr_x + 0.5 * (pred_w - 1.0), pred_ctr_y + 0.5 * (pred_h - 1.0) });
    }
    return retVal;
}
std::vector<double> _get_scores(InferenceEngine::MemoryBlob::Ptr rawData, int anchor_num) {
    auto desc = rawData->getTensorDesc();
    auto sz = desc.getDims();

    size_t restAnchors = sz[1] - anchor_num;
    std::vector<double> retVal(restAnchors*sz[2] * sz[3]);

    LockedMemory<const void> outputMapped = rawData->rmap();
    const float *memPtr = outputMapped.as<float*>();

    for (size_t x = anchor_num; x < sz[1]; x++) {
        for (size_t y = 0; y < sz[2]; y++) {
            for (size_t z = 0; z < sz[3]; z++) {
                retVal[(y*sz[3] + z)*restAnchors + (x - anchor_num)] = memPtr[ (x*sz[2]+y)*sz[3]+z];
            }
        }
    }
    return retVal;
}

std::vector<double> _get_mask_scores(InferenceEngine::MemoryBlob::Ptr rawData, int anchor_num) {
    auto desc = rawData->getTensorDesc();
    auto sz = desc.getDims();

    size_t restAnchors = sz[1] - anchor_num*2;
    std::vector<double> retVal(restAnchors*sz[2] * sz[3]);

    LockedMemory<const void> outputMapped = rawData->rmap();
    const float *memPtr = outputMapped.as<float*>();

    for (size_t x = anchor_num*2; x < sz[1]; x++) {
        for (size_t y = 0; y < sz[2]; y++) {
            for (size_t z = 0; z < sz[3]; z++) {
                retVal[(y*sz[3] + z)*restAnchors + (x - anchor_num*2)] = memPtr[(x*sz[2] + y)*sz[3] + z];
            }
        }
    }
    return retVal;
}


std::vector<cv::Point2f> _get_landmarks(InferenceEngine::MemoryBlob::Ptr rawData, int anchor_num, const std::vector<DetectionPipelineRetinaface::Anchor>& anchors) {
    auto desc = rawData->getTensorDesc();
    auto sz = desc.getDims();

    LockedMemory<const void> outputMapped = rawData->rmap();
    const float *memPtr = outputMapped.as<float*>();

    auto landmark_pred_len = sz[1] / anchor_num;

    std::vector<cv::Point2f> retVal(landmark_pred_len*sz[2] * sz[3]);

    for (int i = 0; i < anchors.size(); i++)
    {
        auto ctrX = anchors[i].getXCenter();
        auto ctrY = anchors[i].getYCenter();
        auto blockWidth = sz[2]*sz[3];
        for (int j = 0; j < DetectionPipelineRetinaface::LANDMARKS_NUM; j++) {
            retVal.emplace_back((float)(memPtr[i + j * 2* blockWidth] * anchors[i].getWidth() + anchors[i].getXCenter()),
                (float)(memPtr[i + (j * 2 + 1)*blockWidth] * anchors[i].getHeight() + anchors[i].getYCenter()));
        }
    }
    return retVal;
}

std::vector<DetectionPipeline::ObjectDesc> DetectionPipelineRetinaface::process_output(PipelineBase::InferenceResult infResult, double scale_x, double scale_y, double face_prob_threshold) {
    std::vector<Anchor> proposals_list;
    std::vector<double> scores_list;
    std::vector<cv::Point2f> landmarks_list;
    std::vector<double> mask_scores_list;
    for (int idx = 0; idx < anchorCfg.size(); idx++) {
        auto s = anchorCfg[idx].stride;
        auto anchors_fpn = _anchors_fpn[s];
        auto anchor_num = anchors_fpn.size();
        auto scores = _get_scores(infResult.outputsData[separateOutputsNames[OT_SCORES][idx]], anchor_num);
        auto bbox_deltas = infResult.outputsData[separateOutputsNames[OT_BBOX][idx]];
        auto sz = bbox_deltas->getTensorDesc().getDims();
        auto height = sz[2];
        auto width = sz[3];

        //--- Creating strided anchors plane
        std::vector<Anchor> anchors(height*width*anchor_num);
        for (int iw = 0; iw < width; iw++) {
            auto sw = iw * s;
            for (int ih = 0; ih < height; ih++) {
                auto sh = ih * s;
                for (int k = 0; k < anchor_num; k++) {
                    Anchor& anc = anchors[(ih*width + iw)*anchor_num + k];
                    anc.left = anchors_fpn[k].left + sw;
                    anc.top = anchors_fpn[k].top + sh;
                    anc.right = anchors_fpn[k].right + sw;
                    anc.bottom = anchors_fpn[k].bottom + sh;
                }
            }
        }

        auto proposals = _get_proposals(bbox_deltas, anchor_num, anchors);
        auto landmarks = _get_landmarks(infResult.outputsData[separateOutputsNames[OT_LANDMARK][idx]], anchor_num, anchors);
        std::vector<double> maskScores;
        if (shouldDetectMasks) {
            maskScores = _get_mask_scores(infResult.outputsData[separateOutputsNames[OT_MASKSCORES][idx]], anchor_num);
        }

/*        auto itp = proposals.begin();
        auto itl = landmarks.begin();
        auto itm = maskScores.begin();
        for (auto its = scores.begin(); its != scores.end(); its++, itp++, itl++) {
            if (*its < face_prob_threshold) {
                its = scores.erase(its);
                itp = proposals.erase(itp);
                itl = landmarks.erase(itl);
                if (shouldDetectMasks) {
                    itm = maskScores.erase(itm);
                }
            }
            if (shouldDetectMasks) {
                itm++;
            }
        }*/

        for (auto& sc : scores)
        {
            if (sc < face_prob_threshold)
            {
                sc = -1;
            }
        }

        if (scores.size()) {
            auto keep = nms(proposals, scores, 0.5);
            proposals_list.reserve(proposals_list.size() + keep.size());
            scores_list.reserve(scores_list.size() + keep.size());
            landmarks_list.reserve(landmarks_list.size() + keep.size());
            for (auto kp : keep) {
                proposals_list.push_back(proposals[kp]);
                scores_list.push_back(scores[kp]);
                landmarks_list.push_back(landmarks[kp]);
                if (shouldDetectMasks) {
                    mask_scores_list.push_back(maskScores[kp]);
                }
            }
        }
    }

    std::vector<DetectionPipeline::ObjectDesc> retVal(scores_list.size());
    for (int i = 0; i < scores_list.size(); i++) {
        DetectionPipeline::ObjectDesc& objDesc = retVal[i];
        objDesc.confidence = (float)scores_list[i];
        objDesc.x = (float)(proposals_list[i].left / scale_x);
        objDesc.y = (float)(proposals_list[i].top / scale_y);
        objDesc.width = (float)((proposals_list[i].right - proposals_list[i].left + 1) / scale_x);
        objDesc.height = (float)((proposals_list[i].bottom - proposals_list[i].top + 1) / scale_y);
        objDesc.labelID = 1;
        objDesc.label = "Face";
        //mask_scores_list = np.reshape(mask_scores_list, -1)

        //landmarks_x_coords = np.array(landmarks_list)[:, : , ::2].reshape(len(landmarks_list), -1) / scale_x
        //landmarks_y_coords = np.array(landmarks_list)[:, : , 1::2].reshape(len(landmarks_list), -1) / scale_y
        //landmarks_regression = [landmarks_x_coords, landmarks_y_coords]
    }
    return retVal;
}
