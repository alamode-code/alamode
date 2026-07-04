/*
 parsephon.h

 Copyright (c) 2014 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <algorithm>
#include <cctype>
#include <fstream>
#include <functional>
#include <locale>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include "pointers.h"

namespace PHON_NS
{
class Input: protected Pointers
{
public:
    Input(class PHON *);

    ~Input();

    void parce_input(int, char **);

    std::string job_title;

    // FILE_FORMAT tag: true (default) routes restart/state files through
    // the unified HDF5 formats; false forces the legacy text files.
    bool use_hdf5_io = true;

    // ALLOW_UNCONVERGED tag: accept renormalized IFC/structure data from an
    // SCPH/QHA state file even when its iterations did not converge
    // (FC2_TEMPERATURE reads and RESTART_SCPH/RESTART_QHA refuse them by
    // default).
    bool allow_unconverged = false;

private:
    std::ifstream ifs_input;
    bool from_stdin;

    int locate_tag(const std::string &);

    void parse_general_vars();

    void parse_analysis_vars(const bool);

    void parse_scph_vars();

    void parse_kappa_vars(const bool);
    void parse_qha_vars();

    void parse_relax_vars();

    void check_relax_vars();

    void parse_initial_strain();

    void parse_initial_displace();

    void parse_cell_parameter();

    void parse_kpoints();

    void get_var_dict(const std::vector<std::string> &, std::map<std::string, std::string> &);

    static void split_str_by_space(const std::string &, std::vector<std::string> &);

    static bool is_endof_entry(const std::string &str);

    template <typename T_to, typename T_from>
    T_to my_cast(T_from const &);

    template <typename T>
    void assign_val(T &, const std::string &, std::map<std::string, std::string>);

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
