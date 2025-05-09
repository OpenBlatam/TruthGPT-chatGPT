#ifndef CREATE_GPT_DATASET_HPP
#define CREATE_GPT_DATASET_HPP

#include <string>

void create_gpt_dataset(const std::string& text_corpus, int chunk_size = 1024, const std::string& data_directory = "data");

#endif // CREATE_GPT_DATASET_HPP
