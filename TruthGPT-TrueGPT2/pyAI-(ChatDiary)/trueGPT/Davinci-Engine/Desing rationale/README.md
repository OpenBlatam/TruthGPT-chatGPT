# Desing Rationale


### Principles

Scalabilty:

Matching efficient



will depend on the specific requirements and constraints of your application.


# Analog

Numpy arrays a q tables are fast. -> recreate in VHDL hardware desing.

# Math

![alt text](https://i.ibb.co/3pzR1BQ/Captura-de-Pantalla-2023-02-17-a-la-s-20-18-13.png)

## Description

high-level overview of how such a system might be structured:

Input data: The system would take in some input data, which could be in the form of text, audio, or video.

Tokenization: The input data would be tokenized using custom hardware designed to efficiently perform text processing tasks such as text segmentation, word tokenization, and part-of-speech tagging.

Encoding: The tokenized data would then be encoded using custom hardware designed to perform high-speed numerical computations, such as matrix multiplication and convolution.

Decoding: The encoded data would be passed through a decoder, which would use custom hardware to reverse the encoding process and output the final result.

Custom engine: The entire system would be powered by a custom engine, which would coordinate the different hardware components and perform optimizations such as pipelining and parallel processing to maximize performance

## Hardware design

Hardware description language: The hardware design needs to be implemented using hardware description languages such as VHDL or Verilog. This involves writing code that describes the functionality and behavior of the hardware.

```

```

module tokenizer(
input clk,         // clock signal
input reset,       // reset signal
input [7:0] text_in,  // input text to be tokenized
output reg [15:0] token_to_id[0:255],  // token to ID mapping
output reg [15:0] id_to_token[0:255],  // ID to token mapping
output reg [15:0] sequence_out[0:255], // token sequence output
output reg [15:0] sequence_len        // length of token sequence output
);

// Define the state machine states
parameter IDLE = 2'b00;
parameter TOKENIZE = 2'b01;
parameter OUTPUT = 2'b10;

// Define the token to ID mapping
localparam TOKEN_TO_ID_SIZE = 8;
localparam TOKEN_TO_ID_MASK = TOKEN_TO_ID_SIZE - 1;
localparam UNK_ID = TOKEN_TO_ID_SIZE;
reg [15:0] token_to_id[TOKEN_TO_ID_SIZE:0];
initial begin
token_to_id[UNK_ID] = 0;   // UNK token is assigned ID 0
token_to_id[' '] = 1;      // Space token is assigned ID 1
token_to_id['\t'] = 2;     // Tab token is assigned ID 2
token_to_id['\n'] = 3;     // Newline token is assigned ID 3
end

// Define the ID to token mapping
localparam ID_TO_TOKEN_SIZE = 8;
localparam ID_TO_TOKEN_MASK = ID_TO_TOKEN_SIZE - 1;
reg [15:0] id_to_token[ID_TO_TOKEN_SIZE:0];
initial begin
id_to_token[0] = UNK_ID;   // ID 0 is assigned UNK token
id_to_token[1] = ' ';      // ID 1 is assigned space token
id_to_token[2] = '\t';     // ID 2 is assigned tab token
id_to_token[3] = '\n';     // ID 3 is assigned newline token
end

// Define the tokenization state machine
reg [1:0] state = IDLE;
reg [7:0] current_char;
reg [7:0] last_char;
reg [15:0] sequence_pos;
reg [15:0] current_token_id;
reg [15:0] sequence_len_reg;
reg [15:0] token_counter;
reg [15:0] char_counter;

// Define the IDLE state
always @(posedge clk) begin
if (reset) begin
state <= IDLE;
end else begin
case (state)
IDLE: begin
if (text_in != 0) begin
last_char <= 0;
sequence_pos <= 0;
sequence_len_reg <= 0;
token_counter <= 0;
char_counter <= 0;
state <= TOKENIZE;
end
end
default: state <= state;
endcase
end
end

// Define the TOKENIZE state
always @(posedge clk) begin
if (reset) begin
state <= IDLE;
end else begin
case (state)
TOKENIZE: begin
current_char <= text_in;
if (current_char == 0) begin
state <= OUTPUT;
end else begin
char_counter <= char_counter + 1;
if (current_char == ' ' || current_char == '\t

Custom HDL

-- Example HDL

-- Data types
type bit is (0, 1);
type byte is array (0 to 7) of bit;
type word is array (0 to 15) of bit;

-- Operators
function and_op (a, b: bit) return bit is
begin
return a and b;
end and_op;

function or_op (a, b: bit) return bit is
begin
return a or b;
end or_op;

function xor_op (a, b: bit) return bit is
begin
return a xor b;
end xor_op;

-- Control structures
procedure if_statement (condition: bit; then_clause, else_clause: procedure) is
begin
if condition then
then_clause;
else
else_clause;
end if;
end if_statement;

procedure for_loop (variable: out bit; start, end_val: natural; body: procedure) is
begin
for i in start to end_val loop
variable := i;
body;
end loop;
end for_loop;

-- Modules
entity adder is
port (
a: in byte;
b: in byte;
cin: in bit;
s: out byte;
cout: out bit
);
end entity adder;

architecture RTL of adder is
signal carry: bit;
begin
s(0) <= a(0) xor b(0) xor cin;
carry <= (a(0) and b(0)) or (a(0) and cin) or (b(0) and cin);
for i in 1 to 7 loop
s(i) <= a(i) xor b(i) xor carry;
carry <= (a(i) and b(i)) or (a(i) and carry) or (b(i) and carry);
end loop;
s(8) <= carry;
cout <= carry;
end RTL;

-- Signals
signal a, b, s: byte;
signal cin, cout: bit;

-- Timing
constant clock_period: time := 10 ns;
attribute clock_frequency: real := 100 MHz;
attribute setup_time: time := 2 ns;
attribute hold_time: time := 1 ns;

-- Verification
test_adder: process
begin
a <= "01010101";
b <= "10101010";
cin <= 0;
wait for clock_period;
assert s = "11111111" and cout = 1
report "Test failed: expected 11111111 with carry out"
severity error;
a <= "00000000";
b <= "00000000";
cin <= 1;
wait for clock_period;
assert s = "00000001" and cout = 0
report "Test failed: expected 00000001 without carry out"
severity error;
wait;
end process;

Write an HDL compiler: Write a compiler that can translate the custom hardware description language into a format that can be processed by the custom engine. This compiler should take into account the unique features of the engine and generate code that can be executed efficiently on the hardware.

-- Example HDL code
entity adder is
port (
a: in std_logic_vector(7 downto 0);
b: in std_logic_vector(7 downto 0);
cin: in std_logic;
s: out std_logic_vector(7 downto 0);
cout: out std_logic
);
end entity adder;

architecture RTL of adder is
signal carry: std_logic;
begin
s(0) <= a(0) xor b(0) xor cin;
carry <= (a(0) and b(0)) or (a(0) and cin) or (b(0) and cin);
for i in 1 to 7 loop
s(i) <= a(i) xor b(i) xor carry;
carry <= (a(i) and b(i)) or (a(i) and carry) or (b(i) and carry);
end loop;
s(8) <= carry;
cout <= carry;
end RTL;

-- Optimized tokenizer
entity
port
in std_logic_vector(7 downto 0)
in std_logic_vector(7 downto 0)
in std_logic
out std_logic_vector(7 downto 0)
out std_logic
end
architecture
signal std_logic
begin
std_logic(0)<=std_logic_vector(0) xor std_logic_vector(0) xor std_logic;
std_logic<=std_logic_vector(0) and std_logic_vector(0) or std_logic_vector(0) and std_logic_vector(0) or std_logic_vector(0) and std_logic;
for integer in integer to integer loop
std_logic(integer)<=std_logic_vector(integer) xor std_logic_vector(integer) xor std_logic;
std_logic<=std_logic_vector(integer) and std_logic_vector(integer) or std_logic_vector(integer) and std_logic or std_logic_vector(integer) and std_logic;
end loop;
std_logic(8)<=std_logic;
std_logic<=std_logic;
end

nce you have designed your HDL, you will need to write a compiler that can translate the HDL into a format that can be processed by the custom engine.

module gaussian_filter(input image[512, 512], output filtered_image[512, 512], input sigma) {
float kernel[3, 3] = {
{0.0625, 0.125, 0.0625},
{0.125, 0.25, 0.125},
{0.0625, 0.125, 0.0625}
};

    for (int i = 1; i < 511; i++) {
        for (int j = 1; j < 511; j++) {
            float sum = 0;
            for (int ki = -1; ki <= 1; ki++) {
                for (int kj = -1; kj <= 1; kj++) {
                    sum += kernel[ki + 1, kj + 1] * image[i + ki, j + kj];
                }
            }
            filtered_image[i, j] = sum;
        }
    }
}

#include <iostream>
#include <vector>
#include <string>

class HDLCompiler {
public:
void compile(std::string source_code) {
// Parse the source code into an abstract syntax tree
ast_ = parse(source_code);

            // Generate intermediate code from the AST
            intermediate_ = generate_intermediate(ast_);

            // Optimize the intermediate code
            optimized_ = optimize(intermediate_);

            // Generate target code
            target_ = generate_target(optimized_);
        }

    private:
        // The abstract syntax tree
        AST ast_;

        // The intermediate code
        std::vector<std::string> intermediate_;

        // The optimized intermediate code
        std::vector<std::string> optimized_;

        // The target code
        std::vector<std::string> target_;

        AST parse(std::string source_code) {
            // Parse the source code into an AST
            AST ast;
            // ...
            return ast;
        }

        std::vector<std::string> generate_intermediate(AST ast) {
            // Generate intermediate code from the AST
            std::vector<std::string> intermediate;
            // ...
            return intermediate;
        }

        std::vector<std::string> optimize(std::vector<std::string> intermediate) {
            // Optimize the intermediate code
            std::vector<std::string> optimized;
            // ...
            return optimized;
        }

        std::vector<std::string> generate_target(std::vector<std::string> optimized) {
            // Generate target code from the optimized intermediate code
            std::vector<std::string> target;
            // ...
            return target;
        }
};

int main() {
HDLCompiler compiler;
compiler.compile("some source code");
return 0;
}

Decentralizing the HDL compiler could involve breaking it down into smaller, more specialized components that can run on separate devices or nodes in a distributed network

## Engine

Using a custom engine can give you more control over the tokenization process, and can be especially useful for languages or domains that require specialized tokenization rules or algorithms

Boyer-Moore algorithm
More algos ?


## Tokenizer

Implement the tokenizer in a lower-level language such as C or Rust, and call it from Python using a language binding. This can improve the performance of the tokenizer and reduce the memory usage, especially if the tokenizer is a performance bottleneck in a larger system
egex engine only tries to match complete words.

Using a pre-trained tokenizer is a great way to optimize tokenization, especially for large datasets or complex languages


Design the hardware accelerator: The hardware accelerator needs to be designed specifically for tokenization. This involves creating a custom architecture that can efficiently tokenize text.

Implement the hardware design: The hardware design needs to be implemented in hardware using FPGAs, ASICs, or other custom hardware designs. This requires specialized knowledge in hardware design and verification.

Integrate the hardware accelerator: The hardware accelerator needs to be integrated with the existing software stack. This involves developing software drivers and APIs that can interact with the hardware accelerator.

Test and optimize the system: The system needs to be tested and optimized to ensure that it performs efficiently and accurately. This requires benchmarking the system and fine-tuning the hardware and software parameters.

Deploy the system: Once the system has been optimized and tested, it can be deployed in production. This requires ensuring that the system is reliable, secure, and scalable.


In terms of the tokenization process, the FPGA-based accelerator can take in the input text, tokenize it and produce the token to ID mappings and the sequence of token IDs. The software stack can then use the hardware accelerator to perform the tokenization and produce the required outputs.

## References:

https://sci-hub.ru/https://ieeexplore.ieee.org/abstract/document/8693206



## Decorder - Encoder

### Description

The encoder and decoder are implemented as classes with their own forward methods that perform the necessary computations. The encoder takes an input tensor and returns the final hidden state of the RNN cell. The decoder takes a hidden state and generates a sequence of outputs, one token at a time.


There is analog of the cell in OSS with hardware in AI ?



