/*
 files.cpp

 Copyright (c) 2014 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#include "files.h"

using namespace ALM_NS;

Files::Files() = default;

Files::~Files() = default;

void Files::init()
{
    //    file_fcs = job_title + ".fcs";
    //    file_hes = job_title + ".hessian";
}

auto Files::set_prefix(const std::string prefix_in) -> void
{
    job_title = prefix_in;
}

auto Files::get_prefix() const -> std::string
{
    return job_title;
}

auto Files::set_datfile_train(const DispForceFile &dat_in) -> void
{
    datfile_train = dat_in;
}

auto Files::set_datfile_validation(const DispForceFile &dat_in) -> void
{
    datfile_validation = dat_in;
}

auto Files::get_datfile_train() const -> DispForceFile
{
    return datfile_train;
}

auto Files::get_datfile_validation() const -> DispForceFile
{
    return datfile_validation;
}
