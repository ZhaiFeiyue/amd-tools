////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2015, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

// #include <hip/hip_runtime.h>
#include <sys/stat.h>

#include <filesystem>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <hip/hip_runtime_api.h>

using namespace std;

int main(int argc, char **argv)
{
    int device_count = 0;
    hipError_t error = hipSuccess;
    error = hipGetDeviceCount(&device_count);
    if (error != hipSuccess)
    {
        std::cout << "no hip device found" << std::endl;
        return -1;
    }
    else
    {
        std::cout << device_count << " devices found " << std::endl;
    }

    std::vector<std::string> bus_ids;
    std::vector<std::string> bus_ids_path;

    for (int i = 0; i < device_count; i++)
    {
        char id[100] = {0};
        error = hipDeviceGetPCIBusId(id, 100, i);
        if (error != hipSuccess)
        {
            std::cout << "get device " << i << " bus failed " << std::endl;
        }
        else
        {
            std::cout << "device " << i << ", bus id = " << id << std::endl;
        }
        bus_ids.push_back(std::string(id));
    }
    std::string device_base_path = "/sys/bus/pci/devices/";
    for (size_t i = 0; i < bus_ids.size(); i++)
    {
        auto tmp = device_base_path + bus_ids[i];

        char real_path[1024] = {0};
        if (realpath(tmp.data(), real_path) == NULL)
        {
            std::cout << "get real path failed " << i << " : " << bus_ids[i] << std::endl;
            continue;
        }
        bus_ids_path.push_back(std::string(real_path));
    }

    std::string nic_base_path = "/sys/class/infiniband/bnxt_re";
    std::vector<std::string> nic_ids;
    std::vector<std::string> nic_ids_path;

    for (size_t i = 0; i < 8; i++)
    {
        std::string tmp = nic_base_path + std::to_string(i);
        char real_path[1024] = {0};
        if (realpath(tmp.data(), real_path) == NULL)
        {
            std::cout << "get real path failed " << i << " : " << tmp << std::endl;
            continue;
        }
        nic_ids_path.push_back(std::string(real_path));
        nic_ids.push_back("bnxt_re" + std::to_string(i));
    }

    if (nic_ids.size() == 0)
    {
        std::cout << "no nic found" << std::endl;
        return -1;
    }

    for (int i = 0; i < device_count; i++)
    {
        size_t distance = 1000;
        size_t index = -1;
        for (size_t j = 0; j < nic_ids_path.size(); j++)
        {
            auto max_len = std::max(nic_ids_path[i].size(), bus_ids_path[i].size());
            for (size_t k = 0; k < max_len; k++)
            {
                if (nic_ids_path[j][k] == bus_ids_path[i][k])
                {
                    max_len--;
                }
                else
                {
                    break;
                }
            }

            if (distance > max_len)
            {
                index = j;
                distance = max_len;
            }
        }
        std::cout << i << ": GPU = " << bus_ids[i] << ", NIC = " << nic_ids[index] << std::endl;
    }

    return 0;
}
