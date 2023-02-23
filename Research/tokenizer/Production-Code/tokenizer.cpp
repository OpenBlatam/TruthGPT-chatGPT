#include <boost/tokenizer.hpp>
#include <vector>
#include <string>

std::vector<std::string> splitString(const std::string& text, const char* separator = " ") {
    boost::char_separator<char> sep(separator);
    boost::tokenizer<boost::char_separator<char>> tokens(text, sep);
    return std::vector<std::string>(tokens.begin(), tokens.end());
}
