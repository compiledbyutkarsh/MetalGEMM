#include "../include/gemm.hpp"
#include "../include/matrix.hpp"
#include <iostream>
#include <iomanip>
#include <string>
#include <cstring>

using namespace MetalGEMM;

static void print_banner() {
    std::cout << "\n";
    std::cout << "  ███╗   ███╗███████╗████████╗ █████╗ ██╗      ██████╗ ███████╗███╗   ███╗███╗   ███╗\n";
    std::cout << "  ████╗ ████║██╔════╝╚══██╔══╝██╔══██╗██║     ██╔════╝ ██╔════╝████╗ ████║████╗ ████║\n";
    std::cout << "  ██╔████╔██║█████╗     ██║   ███████║██║     ██║  ███╗█████╗  ██╔████╔██║██╔████╔██║\n";
    std::cout << "  ██║╚██╔╝██║██╔══╝     ██║   ██╔══██║██║     ██║   ██║██╔══╝  ██║╚██╔╝██║██║╚██╔╝██║\n";
    std::cout << "  ██║ ╚═╝ ██║███████╗   ██║   ██║  ██║███████╗╚██████╔╝███████╗██║ ╚═╝ ██║██║ ╚═╝ ██║\n";
    std::cout << "  ╚═╝     ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚══════╝╚═╝     ╚═╝╚═╝     ╚═╝\n";
    std::cout << "\n";
    std::cout << "          GPU-Accelerated Matrix Multiplication | Metal + Accelerate\n";
    std::cout << "                    github.com/compiledbyutkarsh\n";
    std::cout << "\n";
    std::cout << "================================================================\n";
}

static void print_usage(const char *prog) {
    std::cout << "\n Usage: " << prog << " [options]\n\n";
    std::cout << " Options:\n";
    std::cout << "   -m <M>        Matrix rows (default: 1024)\n";
    std::cout << "   -n <N>        Matrix cols (default: 1024)\n";
    std::cout << "   -k <K>        Inner dimension (default: 1024)\n";
    std::cout << "   -b            Run full benchmark suite\n";
    std::cout << "   -v            Verify correctness\n";
    std::cout << "   -h            Show this help\n\n";
    std::cout << " Examples:\n";
    std::cout << "   " << prog << " -b\n";
    std::cout << "   " << prog << " -m 2048 -n 2048 -k 2048 -b\n";
    std::cout << "   " << prog << " -v\n\n";
}

int main(int argc, char *argv[]) {
    print_banner();

    size_t M = 1024, N = 1024, K = 1024;
    bool run_bench  = false;
    bool run_verify = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) M = std::stoul(argv[++i]);
        else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) N = std::stoul(argv[++i]);
        else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) K = std::stoul(argv[++i]);
        else if (strcmp(argv[i], "-b") == 0) run_bench  = true;
        else if (strcmp(argv[i], "-v") == 0) run_verify = true;
        else if (strcmp(argv[i], "-h") == 0) { print_usage(argv[0]); return 0; }
    }

    if (!run_bench && !run_verify) {
        run_bench  = true;
        run_verify = true;
    }

    GEMMEngine engine;

    if (run_verify) {
        engine.verify_correctness(64,  64,  64);
        engine.verify_correctness(128, 128, 128);
        engine.verify_correctness(256, 256, 256);
    }

    if (run_bench) {
        engine.run_all_benchmarks(256,  256,  256);
        engine.run_all_benchmarks(512,  512,  512);
        engine.run_all_benchmarks(M, N, K);
        if (M == 1024) {
            engine.run_all_benchmarks(2048, 2048, 2048);
        }
    }

    return 0;
}
