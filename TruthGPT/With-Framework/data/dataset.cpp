#ifndef CREATE_GPT_DATASET_HPP
#define CREATE_GPT_DATASET_HPP

#include <deque>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include <zstd.h>
#include <ThreadPool.h>
#include <range/v3/view/chunk.hpp>
#include <range/v3/view/shuffle.hpp>

namespace rv = ranges::views;

void create_gpt_dataset(const std::string& text_corpus, int chunk_size = 1024, const std::string& data_directory = "data", int compression_level = 3, int num_threads = 1) {
    // Create the data directory if it doesn't already exist
    std::filesystem::create_directories(data_directory);

    // Split the text corpus into smaller chunks of text
    auto chunks = text_corpus | rv::chunk(chunk_size) | rv::common;

    // Shuffle the chunks to ensure randomness in the training data
    std::random_device rd;
    std::mt19937 g(rd());
    ranges::shuffle(chunks, g);

    // Compress the chunks in parallel using a thread pool
    ThreadPool pool(num_threads);
    std::deque<std::shared_ptr<std::vector<char>>> compressed_chunks(chunks.size());
    std::vector<std::future<void>> futures;
    for (size_t i = 0; i < chunks.size(); i++) {
        futures.emplace_back(pool.enqueue([&, i] {
            // Compress the chunk using Zstandard
            size_t compressed_size = ZSTD_compressBound(chunk_size);
            std::shared_ptr<std::vector<char>> compressed_chunk = std::make_shared<std::vector<char>>(compressed_size);

            compressed_size = ZSTD_compress(
                compressed_chunk->data(), compressed_size,
                chunks[i].data(), chunk_size,
                compression_level
            );
            compressed_chunk->resize(compressed_size);

            compressed_chunks[i] = compressed_chunk;
        }));
    }

    // Wait for all compression tasks to complete
    for (auto& future : futures) {
        future.wait();
    }

    // Save the compressed chunks to individual files in parallel using a thread pool
    futures.clear();
    for (size_t i = 0; i < compressed_chunks.size(); i++) {
        futures.emplace_back(pool.enqueue([&, i] {
            // Save the compressed chunk to a file
            std::string filename = data_directory + "/chunk_" + std::to_string(i) + ".zst";
            std::ofstream file(filename, std::ios::binary);
            file.write(compressed_chunks[i]->data(), compressed_chunks[i]->size());
        }));
    }

    // Wait for all file I/O tasks to complete
    for (auto& future : futures) {
        future.wait();
    }

    std::cout << "Created " << chunks.size() << " compressed files in directory " << data_directory << "." << std::endl;
}

#endif // CREATE_GPT_DATASET_HPP
