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
#include <map>
#include <string>
#include "gflags/gflags.h"

struct CnnConfig{
    std::string devices;
    std::string cpuExtensionsPath;
    std::string clKernelsConfigPath;
    unsigned int maxAsyncRequests=2;
    std::map < std::string, std::string> execNetworkConfig;
};

class ConfigFactory{
public:
    static CnnConfig GetUserConfig();
    static CnnConfig GetMinLatencyConfig();
protected:
    static CnnConfig GetCommonConfig();
};

