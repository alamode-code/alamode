/*
 input_parser.h

 Copyright (c) 2014 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <algorithm>
#include <cctype>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include "input_setter.h"

namespace PHON_NS
{
class PHON;

class InputParser
{
public:
    InputParser();

    ~InputParser();

    void run(PHON *phon, int narg, const char *const *arg);

    [[nodiscard]] std::string get_run_mode() const;

private:
    std::ifstream ifs_input;
    bool from_stdin = false;

    // Parser state shared between the input blocks (the &general values
    // steer how the later blocks are parsed, and &relax consistency is
    // checked against the &analysis / &scph values).
    std::string job_title;
    std::string run_mode;
    bool use_hdf5_io = true;
    int quartic_mode = 0;
    int relax_str = 0;
    bool calc_FE_bubble = false;
    unsigned int scph_bubble = 0;
    RelaxInputVars relax_vars;

    std::unique_ptr<InputSetter> input_setter;

    void parse_input(PHON *phon);

    void parse_general_vars(PHON *phon);

    void parse_analysis_vars(PHON *phon, bool use_default_values);

    void parse_cell_parameter(PHON *phon);

    void parse_kpoints(PHON *phon);

    void parse_kappa_vars(PHON *phon, bool use_default_values);

    void parse_scph_vars(PHON *phon);

    void parse_qha_vars(PHON *phon);

    void parse_relax_vars(PHON *phon);

    void check_relax_vars() const;

    void parse_initial_strain(PHON *phon);

    void parse_initial_displace(PHON *phon);

    int locate_tag(const std::string &key);

    // Read the payload lines of the current block (comments stripped,
    // blank lines skipped) up to the end-of-entry marker.
    std::vector<std::string> read_block_lines();

    void get_var_dict(const std::vector<std::string> &input_list, std::map<std::string, std::string> &var_dict);

    static void split_str_by_space(const std::string &str, std::vector<std::string> &str_vec);

    static bool is_endof_entry(const std::string &str);

    template <typename T_to, typename T_from>
    T_to my_cast(T_from const &x);

    template <typename T>
    void assign_val(T &val, const std::string &key, std::map<std::string, std::string> dict);

    std::vector<std::string> my_split(const std::string &str, char delim) const
    {
        std::istringstream iss(str);
        std::string str_tmp;
        std::vector<std::string> ret;

        while (std::getline(iss, str_tmp, delim)) {
            ret.push_back(str_tmp);
        }
        return ret;
    }
};

// trim from start
static inline std::string &ltrim(std::string &s)
{
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int c) { return !std::isspace(c); }));
    return s;
}

// trim from end
static inline std::string &rtrim(std::string &s)
{
    s.erase(std::find_if(s.rbegin(), s.rend(), [](int c) { return !std::isspace(c); }).base(), s.end());
    return s;
}

// trim from both ends
static inline std::string &trim(std::string &s)
{
    return ltrim(rtrim(s));
}
} // namespace PHON_NS
