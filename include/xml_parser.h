/*
 xml_parser.h

 Copyright (c) 2014 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <boost/optional.hpp>
#include <boost/property_tree/ptree.hpp>
#include <iostream>
#include <string>

inline auto get_value_from_xml(const boost::property_tree::ptree &pt_in, std::string str,
                               const int exit_when_error = 1) -> std::string
{
    if (boost::optional<std::string> str_entry = pt_in.get_optional<std::string>(str)) {
        return str_entry.get();
    } else {
        if (exit_when_error) {
            std::cout << " Error in get_value_from_xml\n";
            std::cout << " The following entry could not be found in the XML file : " << str << '\n';
            exit(EXIT_FAILURE);
        } else {
            return "";
        }
    }
}
