#include <boost/tokenizer.hpp>
#include <iostream>
#include <string>
#include <vector>

std::vector<std::string> tokenize(const std::string& text) {
    // Create a tokenizer that splits on whitespace and punctuation
    boost::char_separator<char> separator("", " \t\n\r\".,:;()[]{}!?");
    boost::tokenizer<boost::char_separator<char>> tokens(text, separator);

    // Convert the tokens to a vector of strings
    std::vector<std::string> result(tokens.begin(), tokens.end());

    return result;
}

int main() {
    std::string text = "The quick brown fox jumps over the lazy dog.";
    std::vector<std::string> tokens = tokenize(text);

    for (const auto& token : tokens) {
        std::cout << token << std::endl;
    }

    return 0;
}
