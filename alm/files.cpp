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

void Files::set_prefix(const std::string prefix_in)
{
    job_title = prefix_in;
}

std::string Files::get_prefix() const
{
    return job_title;
}

void Files::set_datfile_train(const DispForceFile &dat_in)
{
    datfile_train = dat_in;
}

void Files::set_datfile_validation(const DispForceFile &dat_in)
{
    datfile_validation = dat_in;
}

DispForceFile Files::get_datfile_train() const
{
    return datfile_train;
}

DispForceFile Files::get_datfile_validation() const
{
    return datfile_validation;
}
