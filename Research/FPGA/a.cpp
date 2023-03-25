
/***************************************************************************************
 * Create a Verilog module for a decoder with data_in used as a configuration
 *protocol in FPGA architecture
 *
 *                     Address
 *                   | | ... |
 *                   v v     v
 *                 +-----------+
 *        Enable->/             \<-data_in
 *               /    Decoder    \
 *              +-----------------+
 *                | | | ... | | |
 *                v v v     v v v
 *                    Data output
 *
 *  The outputs are assumes to be one-hot codes (at most only one '1' exist)
 *  Only the data output at the address bit will show data_in
 *
 *  The decoder has an enable signal which is active at logic '1'.
 *  When activated, the decoder will output decoding results to the data output
 *port Otherwise, the data output port will be always all-zero
 ***************************************************************************************/
