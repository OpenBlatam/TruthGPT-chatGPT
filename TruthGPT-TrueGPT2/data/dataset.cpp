#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <filesystem>

#include <range/v3/view/chunk.hpp>
#include <range/v3/view/shuffle.hpp>

#include <fmt/core.h>
#include <docopt/docopt.h>
#include <ThreadPool.h>

namespace fs = std::filesystem;
namespace rv = ranges::views;

void create_gpt_dataset(const std::string& text_corpus, int chunk_size, const std::string& data_directory) {
    fs::create_directories(data_directory);

    auto chunks = text_corpus | rv::chunk(chunk_size) | rv::common;

    std::random_device rd;
    std::mt19937_64 g(rd());
    ranges::shuffle(chunks, g);

    ThreadPool pool(std::thread::hardware_concurrency());
    std::vector<std::future<void>> futures;

    size_t i = 0;
    for (const auto& chunk : chunks) {
        auto filename = fmt::format("{}/chunk_{}.txt", data_directory, i);
        futures.push_back(pool.enqueue([filename, chunk]() {
            std::ofstream file(filename);
            file << chunk;
        }));
        i++;
    }

    for (const auto& future : futures) {
        future.wait();
    }

    std::cout << "Created " << chunks.size() << " files in directory " << data_directory << "." << std::endl;
}

int main(int argc, char** argv) {
    static const char USAGE[] =
    R"(Usage:
      create_gpt_dataset [options] <text_corpus>

    Options:
      -c <chunk_size>, --chunk-size <chunk_size>  The size of each chunk of text [default: 1024].
      -o <output_dir>, --output-dir <output_dir>   The directory to store the data [default: data].
      -h, --help                                  Show this help message and exit.
    )";

    std::map<std::string, docopt::value> args = docopt::docopt(USAGE, { argv + 1, argv + argc });

    const std::string text_corpus = args["<text_corpus>"].asString();
    const int chunk_size = args["--chunk-size"].asLong();
    const std::string output_dir = args["--output-dir"].asString();

    create_gpt_dataset(text_corpus, chunk_size, output_dir);

    return 0;
}
