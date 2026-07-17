#include "Types.h"

#include <unitree/robot/channel/channel_factory.hpp>

#include <boost/program_options.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace po = boost::program_options;

namespace
{
std::atomic_bool keep_running{true};

void handle_signal(int)
{
    keep_running = false;
}

std::filesystem::path executable_path()
{
    std::vector<char> path(1024);
    const auto len = readlink("/proc/self/exe", path.data(), path.size());
    if (len <= 0) {
        return std::filesystem::current_path();
    }
    path[static_cast<std::size_t>(len)] = '\0';
    return std::filesystem::path(path.data());
}

std::filesystem::path default_output_path()
{
    auto exe = executable_path();
    auto project_dir = exe.parent_path();
    if (project_dir.filename() == "build" || project_dir.filename() == "bin") {
        project_dir = project_dir.parent_path();
    }

    const auto now = std::chrono::system_clock::now();
    const auto now_time = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    localtime_r(&now_time, &tm);

    std::ostringstream name;
    name << "lowstate_" << std::put_time(&tm, "%Y%m%d_%H%M%S") << ".csv";
    return project_dir / "log" / name.str();
}

template <typename ArrayType>
void write_array(std::ofstream & csv, const ArrayType & data)
{
    for (const auto & value : data) {
        csv << "," << value;
    }
}

void write_header(std::ofstream & csv)
{
    csv << "time,tick";

    for (int i = 0; i < 12; ++i) {
        csv << ",q" << i;
    }
    for (int i = 0; i < 12; ++i) {
        csv << ",dq" << i;
    }
    for (int i = 0; i < 12; ++i) {
        csv << ",tau" << i;
    }
    for (int i = 0; i < 12; ++i) {
        csv << ",temp" << i;
    }
    for (int i = 0; i < 12; ++i) {
        csv << ",lost" << i;
    }

    csv << ",imu_qw,imu_qx,imu_qy,imu_qz";
    csv << ",gyro_x,gyro_y,gyro_z";
    csv << ",acc_x,acc_y,acc_z";
    csv << ",roll,pitch,yaw";
    csv << ",foot_force0,foot_force1,foot_force2,foot_force3";
    csv << ",foot_force_est0,foot_force_est1,foot_force_est2,foot_force_est3";
    csv << ",joy_lx,joy_ly,joy_rx,joy_ry";
    csv << ",joy_cmd_vx_raw,joy_cmd_vy_raw,joy_cmd_yaw_raw";
    csv << "\n";
}

void write_sample(std::ofstream & csv, LowState_t & lowstate, double elapsed)
{
    lowstate.update();

    float joy_lx = lowstate.joystick.lx();
    float joy_ly = lowstate.joystick.ly();
    float joy_rx = lowstate.joystick.rx();
    float joy_ry = lowstate.joystick.ry();

    std::lock_guard<std::mutex> lock(lowstate.mutex_);
    const auto & msg = lowstate.msg_;
    const auto & imu = msg.imu_state();
    const auto & motors = msg.motor_state();

    csv << std::fixed << std::setprecision(6) << elapsed;
    csv << "," << msg.tick();

    for (int i = 0; i < 12; ++i) {
        csv << "," << motors[i].q();
    }
    for (int i = 0; i < 12; ++i) {
        csv << "," << motors[i].dq();
    }
    for (int i = 0; i < 12; ++i) {
        csv << "," << motors[i].tau_est();
    }
    for (int i = 0; i < 12; ++i) {
        csv << "," << static_cast<int>(motors[i].temperature());
    }
    for (int i = 0; i < 12; ++i) {
        csv << "," << motors[i].lost();
    }

    write_array(csv, imu.quaternion());
    write_array(csv, imu.gyroscope());
    write_array(csv, imu.accelerometer());
    write_array(csv, imu.rpy());
    write_array(csv, msg.foot_force());
    write_array(csv, msg.foot_force_est());

    csv << "," << joy_lx << "," << joy_ly << "," << joy_rx << "," << joy_ry;
    csv << "," << joy_ly << "," << -joy_lx << "," << -joy_rx;
    csv << "\n";
}
} // namespace

int main(int argc, char ** argv)
{
    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

    std::string network;
    std::string output;
    std::string topic;
    int domain;
    double duration;
    double rate;

    po::options_description desc("Go2 lowstate logger");
    desc.add_options()
        ("help,h", "produce help message")
        ("domain,D", po::value<int>(&domain)->default_value(0), "dds domain id")
        ("network,n", po::value<std::string>(&network)->default_value(""), "dds network interface")
        ("output,o", po::value<std::string>(&output)->default_value(default_output_path().string()), "output csv path")
        ("topic,t", po::value<std::string>(&topic)->default_value("rt/lowstate"), "lowstate topic")
        ("duration,d", po::value<double>(&duration)->default_value(0.0), "recording duration in seconds; 0 means until Ctrl-C")
        ("rate,r", po::value<double>(&rate)->default_value(200.0), "sampling rate in Hz");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }
    if (rate <= 0.0) {
        std::cerr << "rate must be positive.\n";
        return 1;
    }

    auto output_path = std::filesystem::path(output);
    if (output_path.has_parent_path()) {
        std::filesystem::create_directories(output_path.parent_path());
    }

    unitree::robot::ChannelFactory::Instance()->Init(domain, network);

    auto lowstate = std::make_shared<LowState_t>(topic);
    std::cout << "Waiting for " << topic << "..." << std::endl;
    auto wait_start = std::chrono::steady_clock::now();
    auto last_warn = wait_start;
    while (keep_running && lowstate->isTimeout()) {
        const auto now = std::chrono::steady_clock::now();
        if (now - last_warn > std::chrono::seconds(2)) {
            std::cout << "Still waiting for " << topic << "..." << std::endl;
            last_warn = now;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (!keep_running) {
        std::cout << "Interrupted while waiting for " << topic << "." << std::endl;
        return 130;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    std::cout << "Connected. Writing " << output_path << std::endl;

    std::ofstream csv(output_path);
    if (!csv.is_open()) {
        std::cerr << "Failed to open " << output_path << "\n";
        return 1;
    }
    csv << std::setprecision(9);
    write_header(csv);

    const auto start = std::chrono::steady_clock::now();
    const auto sample_period = std::chrono::duration<double>(1.0 / rate);
    auto next_sample = start;
    std::size_t sample_count = 0;

    while (keep_running) {
        const auto now = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(now - start).count();
        if (duration > 0.0 && elapsed >= duration) {
            break;
        }

        write_sample(csv, *lowstate, elapsed);
        ++sample_count;
        if (sample_count % 200 == 0) {
            csv.flush();
        }

        next_sample += std::chrono::duration_cast<std::chrono::steady_clock::duration>(sample_period);
        std::this_thread::sleep_until(next_sample);
    }

    csv.flush();
    std::cout << "Saved " << sample_count << " samples to " << output_path << std::endl;
    return 0;
}
