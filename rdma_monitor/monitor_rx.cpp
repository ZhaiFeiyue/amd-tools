#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>
#include <fcntl.h>
#include <thread>
#include <fstream>
#include <cstdlib>
#include <csignal>
#include <atomic>
#include <ranges>
#include <numeric>
#include "utils.h"

std::atomic<bool> exit_requested(false);

using namespace std::literals::chrono_literals;
namespace fs = std::filesystem;
#define NOW (std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count())
#define MAX_SAMPLE (2000000)
#define SAMPLE_INTERVAL (1000) // ns

struct RawProfilePoint{
    uint64_t timestamp;
    uint64_t bytes;
    double bw; //gbs
};


void signal_handler(int signal) {
    if (signal == SIGINT) {
        std::cout << "\n saving file... \n" << std::endl;
        exit_requested = true;
    }
}

void profile_nic_rx(std::string nic_name){
    auto path = std::string("/sys/class/infiniband/") + nic_name + std::string("/ports/1/hw_counters/rx_bytes");
    std::vector<uint64_t> bytes;
    std::vector<uint64_t> ts;
    bytes.reserve(MAX_SAMPLE);
    ts.reserve(MAX_SAMPLE);
    for(int i=0;i<MAX_SAMPLE;i++){
        bytes.push_back(0);
        ts.push_back(0);
    }
    
    
    char read_buffer[32] = {0};
    int current_time;
    int counts = 0;
    while(true && !exit_requested){
        auto start = NOW;
        while (true) {
            if(NOW - start > 1000){
                std::ifstream ifs{path};
                ifs.read(read_buffer,32-1);
                ifs.close();
                bytes[counts] = std::strtoull(read_buffer, nullptr, 10);
                ts[counts] = NOW;
                break;
            }
        }
        if(counts >= MAX_SAMPLE){break;}
        counts++;
        std::this_thread::sleep_for(10us);
    }
    std::string file_name = std::string("data_")+nic_name+std::string("_rx.txt");
    std::fstream ofs(file_name, std::ios::out | std::ios::trunc);
    if(ofs){
        for(int i=0;i<MAX_SAMPLE;i++){
            ofs<<ts[i]<<","<<bytes[i]<<"\n";
        }
    }
    ofs.close();

    // 对于已经写入的数据文件,删除其中重复的rx列的数据
    std::ifstream ifs(file_name);
    if (!ifs.is_open()) {
        std::cerr << "Error: Failed to open file " << file_name << std::endl;
        throw std::runtime_error("Error: Failed to open file " + file_name);
    }

    std::vector<RawProfilePoint> profile_data;
    std::string line;
    while(std::getline(ifs,line)){
        std::istringstream ss(line);
        uint64_t timestamp;
        uint64_t bytes;
        char comma;
        if (ss >> timestamp >> comma >> bytes) {
            if (comma == ',') {
                profile_data.push_back(RawProfilePoint{.timestamp = timestamp,.bytes=bytes});
            }
        }
        else if (!line.empty()) {
            std::cerr << "Warning: Skipping malformed line: " << line << std::endl;
        }
    }
    ifs.close();

    uint64_t prev_bytes = 0;
    uint64_t cur_bytes = 0;
    uint64_t timestamp0 = 0;
    for(auto& point:profile_data){
        if(point.bytes == 0){
            break;
        }
        if(cur_bytes == point.bytes){
            continue;
        }
        else{
            prev_bytes = cur_bytes;
            cur_bytes = point.bytes;
            double gbs = (cur_bytes - prev_bytes) / 1.0 / (point.timestamp - timestamp0);
            timestamp0 = point.timestamp;
            point.bw = gbs;
        }
    }
    auto start_timestamp = profile_data[0].timestamp;

    auto filtered = profile_data | std::views::filter([](RawProfilePoint point){return point.bw > 0.00000001;})
                                | std::views::transform([start_timestamp](RawProfilePoint point){point.timestamp -= start_timestamp;return point;});

    // 处理后的数据写入文件方便使用python制作成perfetto文件或者可视化
    std::string processed_peofile_file_name = std::string("data_")+nic_name+std::string("_rx_processed.txt");
    std::fstream ofs2(processed_peofile_file_name, std::ios::out | std::ios::trunc);
    if(ofs){
        for(auto point:filtered){
            ofs2<<point.timestamp<<","<<point.bw<<"\n";
        }
    }
    ofs2.close();


    // (TODO.)打印一下更多summary信息
    utils::println("收集数据已经存储在:",processed_peofile_file_name,"可以使用python做可视化");
}

int main() {
    signal(SIGINT,signal_handler);
    std::vector<std::thread> threads;
    auto nics = utils::map([](fs::directory_entry entry){return entry.path().filename().string();}, 
                        std::vector<fs::directory_entry>{fs::directory_iterator("/sys/class/infiniband/"),
                                fs::directory_iterator{}});
    utils::println("NICS:",nics);

    for(auto nic_name:nics){
        threads.emplace_back(profile_nic_rx,nic_name);
    }

    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }


}
