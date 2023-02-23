#include <torch/torch.h>
#include <torch/script.h>
#include <fstream>
#include <vector>
#include <sstream>
#include <memory>
#include <mio/mmap.hpp>

struct ModelData {
    std::shared_ptr<torch::jit::script::Module> model;
    std::shared_ptr<torch::jit::script::Module> tokenizer;
};

ModelData load_model(const std::string& model_path, const std::string& tokenizer_path) {
    mio::mmap_source model_file(model_path);
    auto model = torch::jit::load(model_file.data(), model_file.size());
    auto model_ptr = std::make_shared<torch::jit::script::Module>(std::move(model));

    mio::mmap_source tokenizer_file(tokenizer_path);
    auto tokenizer = torch::jit::load(tokenizer_file.data(), tokenizer_file.size());
    auto tokenizer_ptr = std::make_shared<torch::jit::script::Module>(std::move(tokenizer));

    return {model_ptr, tokenizer_ptr};
}

std::vector<int64_t> tokenize(const std::shared_ptr<torch::jit::script::Module>& tokenizer, const std::string& text) {
    auto tokens = tokenizer->run_method("encode", text).toTensor();
    return std::vector<int64_t>(tokens.data_ptr<int64_t>(), tokens.data_ptr<int64_t>() + tokens.numel());
}

std::string generate_text(const std::shared_ptr<torch::jit::script::Module>& model, const std::vector<int64_t>& tokens, int max_length) {
    static const int BATCH_SIZE = 1;
    static const int NUM_LAYERS = 12;
    static const int HIDDEN_SIZE = 768;

    auto hidden = torch::zeros({BATCH_SIZE, NUM_LAYERS, HIDDEN_SIZE});
    model->to(torch::kCPU);
    model->eval();

    auto generated_tokens = tokens;
    generated_tokens.reserve(max_length);
    while (generated_tokens.size() < max_length) {
        auto input_tensor = torch::from_blob(generated_tokens.data(), {BATCH_SIZE, generated_tokens.size()}, torch::kLong);
        auto output = model->forward({input_tensor, hidden}).toTensor();
        auto next_token = output.argmax(-1)[{0, -1}];

        if (next_token.item<int64_t>() == 50256) {
            break;
        }

        generated_tokens.push_back(next_token.item<int64_t>());
    }

    std::stringstream ss;
    std::copy(generated_tokens.begin(), generated_tokens.end(), std::ostream_iterator<int64_t>(ss, " "));
    return ss.str();
}

class TextGenerator {
public:
    TextGenerator(const std::string& model_path, const std::string& tokenizer_path) : model_data_(load_model(model_path, tokenizer_path)) {}

    std::string generate(const std::string& input_text, int max_length) {
        auto tokens = tokenize(model_data_.tokenizer, input_text);
        return generate_text(model_data_.model, tokens, max_length);
    }

private:
    ModelData model_data_;
};

int main(int argc, const char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <model-path> <tokenizer-path>\n";
        return 1;
    }

    TextGenerator generator(argv[1], argv[2]);

    std::string input_text;
    while (true) {
        std::cout << "Enter input text: ";
        std::getline(std::cin, input_text);
        if (input_text.empty()) {
        break;
            }

            std::cout << "Enter maximum length of generated text: ";
            int max_length;
            std::cin >> max_length;

            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

            std::string generated_text = generator.generate(input_text, max_length);
            std::cout << "Generated text: " << generated_text << std::endl;
        }

        return 0;
}
