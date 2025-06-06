/*
 almcui.h

 Copyright (c) 2014, 2015, 2016 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/


#pragma once

namespace ALM_NS
{
class ALMCUI
{
public:
    ALMCUI();

    ~ALMCUI();

    auto run(const int narg, char **arg) const -> void;
};
} // namespace ALM_NS
