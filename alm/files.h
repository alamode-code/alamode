/*
 files.h

 Copyright (c) 2014 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <string>

namespace ALM_NS
{
class DispForceFile
{
public:
    std::string filename;
    size_t ndata, nstart, nend;
    size_t skip_s, skip_e;

    DispForceFile()
    {
        filename = "";
        ndata = 0;
        nstart = 0;
        nend = 0;
        skip_s = 0;
        skip_e = 0;
    }

    ~DispForceFile() = default;

    DispForceFile(const DispForceFile &obj) = default;

    DispForceFile &operator=(const DispForceFile &obj) = default;
};

class Files
{
public:
    Files();

    ~Files();

    void init();

    void set_prefix(const std::string);

    [[nodiscard]] std::string get_prefix() const;

    void set_datfile_train(const DispForceFile &dat_in);

    void set_datfile_validation(const DispForceFile &dat_in);

    [[nodiscard]] DispForceFile get_datfile_train() const;

    [[nodiscard]] DispForceFile get_datfile_validation() const;

private:
    std::string job_title;
    DispForceFile datfile_train, datfile_validation;
};
}
