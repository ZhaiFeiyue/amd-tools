#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <unistd.h>
#include <fcntl.h>
#include <thread>
#include <fstream>
#include <cstdlib>
#include <csignal>
#include <atomic>
#include "utils.h"

std::atomic<bool> exit_requested(false);

using namespace std::literals::chrono_literals;
namespace fs = std::filesystem;
#define NOW (std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count())
#define MAX_SAMPLE (10000)
#define SAMPLE_INTERVAL (1000) // ns

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
    std::fstream ofs(std::string("data_")+nic_name+std::string("_rx.txt"), std::ios::out | std::ios::trunc);
    if(ofs){
        for(int i=0;i<MAX_SAMPLE;i++){
            ofs<<ts[i]<<","<<bytes[i]<<"\n";
        }
    }
    ofs.close();
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
