#pragma once

#include <torch/torch.h>
#include <torch/script.h>
#include <fstream>
#include <vector>
#include <sstream>
#include <memory>
#include <mio/mmap.hpp>

class TextGenerator {
public:
    TextGenerator(const std::string& model_path, const std::string& tokenizer_path);

    std::string generate(const std::string& input_text, int max_length);

private:
    struct ModelData {
        std::shared_ptr<torch::jit::script::Module> model;
        std::shared_ptr<torch::jit::script::Module> tokenizer;
    };

    ModelData load_model(const std::string& model_path, const std::string& tokenizer_path);
    std::vector<int64_t> tokenize(const std::shared_ptr<torch::jit::script::Module>& tokenizer, const std::string& text);
    std::string generate_text(const std::shared_ptr<torch::jit::script::Module>& model, const std::vector<int64_t>& tokens, int max_length);

    ModelData model_data_;
};
