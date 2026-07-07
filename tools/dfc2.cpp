/*
 dfc2.cpp

Copyright (c) 2016 Terumasa Tadano

This file is distributed under the terms of the MIT license.
Please see the file 'LICENCE.txt' in the root directory
or http://opensource.org/licenses/mit-license.php for information.
*/

#include <Eigen/Core>
#include <boost/algorithm/string.hpp>
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/version.hpp>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>

#include "constants.h"
#include "dfc2.h"
#include "mathfunctions.h"
#include "memory.h"
#include "xml_parser.h"

using namespace std;

int main(int argc, char *argv[])
{
    if (argc == 1) {
        cout << " DFC2 -- a generator of renormalized harmonic FCs from SCPH outputs." << endl;
        cout << " XML file containing original FC2 : ";
        cin >> original_xml;
        cout << " Output xml filename with anharmonic correction : ";
        cin >> new_xml;
        cout << " FC2 correction file from SCPH calculation : ";
        cin >> file_fc2_correction;
        cout << " Target temperature : ";
        cin >> temp;
    } else if (argc == 5) {
        original_xml = argv[1];
        new_xml = argv[2];
        file_fc2_correction = argv[3];
        temp = boost::lexical_cast<float>(argv[4]);
    } else {
        std::cout << "Usage: " << std::endl;
        std::cout << "(command line) > dfc2 FC2XML_ORIG FC2XML_NEW SCPH_DFC2 TEMPERATURE" << std::endl;
        std::cout << "(interactive) > dfc2 " << std::endl;
        std::cout << std::endl;
    }

    // Load original harmonic force constants and structure data of the supercell
    load_fc2_xml(original_xml);

    // Load anharmonic correction and structure data of the primitive lattice
    load_delta_fc2(file_fc2_correction, temp);

    // Initialize new fc2 by the original values
    fc2_new.clear();
    copy(fc2_orig.begin(), fc2_orig.end(), back_inserter(fc2_new));

    // Add delta_fc2 to fc2_new
    calculate_new_fc2(fc2_orig, delta_fc2, fc2_new);

    write_new_xml(fc2_new, new_xml);

    cout << endl << " New XML file " << new_xml << " was created successfully." << endl;

    deallocate(xr_s);
    deallocate(kd);
    deallocate(kd_symbol);
    deallocate(map_p2s);
    deallocate(map_s2p);
    deallocate(xr_p);
    deallocate(kd_p);
}


void load_fc2_xml(const std::string file_in)
{
    int i;
    using namespace boost::property_tree;

    ptree pt;
    int atm1, atm2, xyz1, xyz2, cell_s;
    stringstream ss1, ss2;
    FcsClassExtent fcext_tmp;

    map<string, int> dict_atomic_kind;

    try {
        read_xml(file_in, pt);
    } catch (exception &e) {
        cout << "Cannot open file " + file_in << endl;
        exit(EXIT_FAILURE);
    }

    nat = boost::lexical_cast<unsigned int>(get_value_from_xml(pt, "Data.Structure.NumberOfAtoms"));
    nkd = boost::lexical_cast<unsigned int>(get_value_from_xml(pt, "Data.Structure.NumberOfElements"));

    ntran = boost::lexical_cast<unsigned int>(get_value_from_xml(pt, "Data.Symmetry.NumberOfTranslations"));

    natmin = nat / ntran;

    for (i = 0; i < 3; ++i) {
        ss1.str("");
        ss1.clear();
        ss1 << get_value_from_xml(pt, "Data.Structure.LatticeVector.a" + std::to_string(i + 1));
        ss1 >> lavec_s[0][i] >> lavec_s[1][i] >> lavec_s[2][i];
    }

    ss1.str("");
    ss1.clear();
    ss1 << get_value_from_xml(pt, "Data.Structure.Periodicity");
    ss1 >> is_periodic[0] >> is_periodic[1] >> is_periodic[2];

    // Parse atomic elements and coordinates

    allocate(xr_s, nat, 3);
    allocate(kd, nat);
    allocate(kd_symbol, nkd);

    i = 0;

    BOOST_FOREACH (const ptree::value_type &child_, pt.get_child("Data.Structure.AtomicElements")) {
        const ptree &child = child_.second;
        const auto icount_kd = child.get<unsigned int>("<xmlattr>.number");
        dict_atomic_kind[boost::lexical_cast<string>(child_.second.data())] = icount_kd - 1;
        kd_symbol[i++] = boost::lexical_cast<string>(child_.second.data());
    }

    unsigned int index;

    BOOST_FOREACH (const ptree::value_type &child_, pt.get_child("Data.Structure.Position")) {
        const ptree &child = child_.second;
        const string str_index = child.get<string>("<xmlattr>.index");
        const string str_element = child.get<string>("<xmlattr>.element");

        ss1.str("");
        ss1.clear();
        ss1 << child.data();

        index = boost::lexical_cast<unsigned int>(str_index) - 1;

        if (index >= nat) {
            cout << "index is out of range" << endl;
            exit(EXIT_FAILURE);
        }

        kd[index] = dict_atomic_kind[str_element];
        ss1 >> xr_s[index][0] >> xr_s[index][1] >> xr_s[index][2];
    }

    dict_atomic_kind.clear();

    // Parse mapping information

    allocate(map_p2s, natmin, ntran);
    allocate(map_s2p, nat);

    unsigned int tran, atom_p, atom_s;

    BOOST_FOREACH (const ptree::value_type &child_, pt.get_child("Data.Symmetry.Translations")) {
        const ptree &child = child_.second;
        const string str_tran = child.get<string>("<xmlattr>.tran");
        const string str_atom = child.get<string>("<xmlattr>.atom");

        tran = boost::lexical_cast<unsigned int>(str_tran) - 1;
        atom_p = boost::lexical_cast<unsigned int>(str_atom) - 1;
        atom_s = boost::lexical_cast<unsigned int>(child.data()) - 1;

        if (tran >= ntran || atom_p >= natmin || atom_s >= nat) {
            cout << "index is out of range" << endl;
            exit(EXIT_FAILURE);
        }

        map_p2s[atom_p][tran] = atom_s;
        map_s2p[atom_s].atom_num = atom_p;
        map_s2p[atom_s].tran_num = tran;
    }


    BOOST_FOREACH (const ptree::value_type &child_, pt.get_child("Data.ForceConstants.HARMONIC")) {
        const ptree &child = child_.second;
        const string str_p1 = child.get<string>("<xmlattr>.pair1");
        const string str_p2 = child.get<string>("<xmlattr>.pair2");

        ss1.str("");
        ss2.str("");
        ss1.clear();
        ss2.clear();

        ss1 << str_p1;
        ss2 << str_p2;

        ss1 >> atm1 >> xyz1;
        ss2 >> atm2 >> xyz2 >> cell_s;

        fcext_tmp.atm1 = atm1 - 1;
        fcext_tmp.xyz1 = xyz1 - 1;
        fcext_tmp.atm2 = atm2 - 1;
        fcext_tmp.xyz2 = xyz2 - 1;
        fcext_tmp.cell_s = cell_s - 1;
        fcext_tmp.fcs_val = boost::lexical_cast<double>(child.data());

        fc2_orig.push_back(fcext_tmp);
    }
}


void load_delta_fc2(const std::string file_in, const double temp)
{
    int i;
    ifstream ifs_in;
    stringstream ss;

    // Restart
    ifs_in.open(file_in.c_str(), ios::in);
    if (!ifs_in) {
        cout << "Could not open " + file_in << endl;
        exit(EXIT_FAILURE);
    }

    // Check the consistency

    string line_tmp, str_tmp;
    vector<string> str_vec;

    int sx, sy, sz;
    int atm1, atm2, xyz1, xyz2;
    double dfc2_tmp;
    bool found_tag = false;

    // Get lattice vectors
    for (i = 0; i < 3; ++i) {
        ifs_in >> lavec_p[0][i] >> lavec_p[1][i] >> lavec_p[2][i];
    }
    recips(lavec_p, rlavec_p);

    ifs_in >> nat_p >> nkd_p;
    ifs_in.ignore();
    getline(ifs_in, line_tmp);

    allocate(xr_p, nat_p, 3);
    allocate(kd_p, nat_p);
    for (i = 0; i < nat_p; ++i) {
        ifs_in >> xr_p[i][0] >> xr_p[i][1] >> xr_p[i][2] >> kd_p[i];
    }
    ifs_in.ignore();

    while (getline(ifs_in, line_tmp)) {
        if (line_tmp[0] == '#') {
            boost::split(str_vec, line_tmp, boost::is_space());
            if (abs(boost::lexical_cast<double>(str_vec[3]) - temp) < eps) {
                found_tag = true;
                break;
            }
        }
    }
    if (!found_tag) {
        cout << "Could not find the # Temp tag for the target temperature" << endl;
        exit(EXIT_FAILURE);
    }

    delta_fc2.clear();

    while (getline(ifs_in, line_tmp)) {

        if (line_tmp[0] == '#') break;

        if (!line_tmp.empty()) {
            stringstream ss1;

            ss1 << line_tmp;
            ss1 >> sx >> sy >> sz >> atm1 >> xyz1 >> atm2 >> xyz2 >> dfc2_tmp;

            delta_fc2.emplace_back(DeltaFcs(sx, sy, sz, atm1, xyz1, atm2, xyz2, dfc2_tmp));
        }
    }

    ifs_in.close();
}


void calculate_new_fc2(const std::vector<FcsClassExtent> &fc2_in, std::vector<DeltaFcs> delta_fc2,
                       std::vector<FcsClassExtent> &fc2_out)
{
    int i, j, k;
    int ix, iy, iz;
    int icell;
    double **xshift_s;

    allocate(xshift_s, 27, 3);

    for (i = 0; i < 3; ++i) xshift_s[0][i] = 0.0;

    icell = 0;

    for (ix = -1; ix <= 1; ++ix) {
        for (iy = -1; iy <= 1; ++iy) {
            for (iz = -1; iz <= 1; ++iz) {
                if (ix == 0 && iy == 0 && iz == 0) continue;

                ++icell;

                xshift_s[icell][0] = static_cast<double>(ix);
                xshift_s[icell][1] = static_cast<double>(iy);
                xshift_s[icell][2] = static_cast<double>(iz);
            }
        }
    }

    double mat_convert[3][3];


    for (i = 0; i < 3; ++i) {
        for (j = 0; j < 3; ++j) {
            mat_convert[i][j] = 0.0;
            for (k = 0; k < 3; ++k) {
                mat_convert[i][j] += rlavec_p[i][k] * lavec_s[k][j] / (2.0 * pi);
            }

            mat_convert[i][j] = static_cast<double>(nint(mat_convert[i][j]));
        }
    }

    double vec[3];

    int icount = 0;

    vector<int> arr_tmp;
    vector<FcsTrans> fc2_data;

    fc2_data.clear();

    for (const auto &it: fc2_in) {

        arr_tmp.clear();

        for (i = 0; i < 3; ++i) {
            vec[i] = xr_s[it.atm2][i] + xshift_s[it.cell_s][i] - xr_s[map_p2s[map_s2p[it.atm2].atom_num][0]][i];
        }

        rotvec(vec, vec, mat_convert);

        arr_tmp.push_back(it.atm1);
        arr_tmp.push_back(it.xyz1);
        arr_tmp.push_back(map_s2p[it.atm2].atom_num);
        arr_tmp.push_back(it.xyz2);
        for (i = 0; i < 3; ++i) arr_tmp.push_back(nint(vec[i]));

        //        std::cout << std::setw(5) << vec[0];
        //        std::cout << std::setw(5) << vec[1];
        //        std::cout << std::setw(5) << vec[2];
        //
        //        std::cout << std::setw(5) << it.atm1 << std::setw(5) << it.xyz1;
        //        std::cout << std::setw(5) << map_super_to_trueprim[it.atm2].atom_num << std::setw(5) << it.xyz2;
        //        std::cout << std::setw(15) << it.fcs_val << '\n';

        fc2_data.emplace_back(arr_tmp, icount);
        ++icount;
    }

    std::sort(fc2_data.begin(), fc2_data.end());

    vector<FcsTrans>::iterator iter_found;
    int index_tmp = 0;

    bool detect_warnings = false;

    for (auto &it: delta_fc2) {

        if (abs(it.dfc2) > eps10) {
            arr_tmp.clear();
            arr_tmp.push_back(it.atm1);
            arr_tmp.push_back(it.xyz1);
            arr_tmp.push_back(it.atm2);
            arr_tmp.push_back(it.xyz2);
            arr_tmp.push_back(it.sx);
            arr_tmp.push_back(it.sy);
            arr_tmp.push_back(it.sz);

            iter_found = lower_bound(fc2_data.begin(), fc2_data.end(), FcsTrans(arr_tmp, index_tmp));

            if (iter_found != fc2_data.end() && arr_tmp == (*iter_found).arr) {
                fc2_new[(*iter_found).fcs_index].fcs_val += it.dfc2;
            } else {
                cout << "Warning: The following force constant doesn't exist in the original file:" << endl;
                cout << setw(5) << it.sx << setw(5) << it.sy << setw(5) << it.sz;
                cout << setw(5) << it.atm1 << setw(5) << it.xyz1;
                cout << setw(5) << it.atm2 << setw(5) << it.xyz2;
                cout << setw(15) << it.dfc2 << endl;
                detect_warnings = true;
            }
        }
        ++index_tmp;
    }

    if (detect_warnings) {
        cout << "\n";
        cout << " If you see a warning message above, please check the following points:\n";
        cout << " 0. If the value in the last column is small enough (e.g., < 1.0e-8), \n"
                "    the warning message usually does not signify an error of your calculation.\n";
        cout << " 1. If the value in the last collumn is large enough, please make sure that the q-point grid\n"
                "    specified by KMESH_INTERPOLATE (e.g., NxMxL) is smaller than or equal to the supercell size\n"
                "    of the original FC2. If this is not the case, please change KMESH_INTERPOLATE or\n"
                "    the input XML file containing the original FC2.\n";
        cout << " 2. If the above condition 1 is satisifed but you still get a warning,\n"
                "    please check if the ANPHON code detected the space group consistently with ALM.\n"
                "    If not, please change TOLERANCE value and/or refine the input structure.\n";
        cout << " 3. If the harmonic force constants are extracted for a model potential where\n"
                "    the absolute value of force constants become very small (<1.0e-12) beyond\n"
                "    a certain cutoff radius inside ths supercell, please use a smaller FC_ZERO_THR\n"
                "    when computing harmonic force constants. This may solve the problem.\n";
        cout << " 4. If you still see a warning after checking the above points,\n"
                "    please report it at https://github.com/ttadano/alamode/issues.\n";
    }
}


void recips(double vec[3][3], double inverse[3][3])
{
    double det;
    det = vec[0][0] * vec[1][1] * vec[2][2] + vec[1][0] * vec[2][1] * vec[0][2] + vec[2][0] * vec[0][1] * vec[1][2] -
          vec[0][0] * vec[2][1] * vec[1][2] - vec[2][0] * vec[1][1] * vec[0][2] - vec[1][0] * vec[0][1] * vec[2][2];

    if (abs(det) < eps12) {
        cout << "Lattice vector is singular" << endl;
        exit(EXIT_FAILURE);
    }

    double factor = 2.0 * pi / det;

    inverse[0][0] = (vec[1][1] * vec[2][2] - vec[1][2] * vec[2][1]) * factor;
    inverse[0][1] = (vec[0][2] * vec[2][1] - vec[0][1] * vec[2][2]) * factor;
    inverse[0][2] = (vec[0][1] * vec[1][2] - vec[0][2] * vec[1][1]) * factor;

    inverse[1][0] = (vec[1][2] * vec[2][0] - vec[1][0] * vec[2][2]) * factor;
    inverse[1][1] = (vec[0][0] * vec[2][2] - vec[0][2] * vec[2][0]) * factor;
    inverse[1][2] = (vec[0][2] * vec[1][0] - vec[0][0] * vec[1][2]) * factor;

    inverse[2][0] = (vec[1][0] * vec[2][1] - vec[1][1] * vec[2][0]) * factor;
    inverse[2][1] = (vec[0][1] * vec[2][0] - vec[0][0] * vec[2][1]) * factor;
    inverse[2][2] = (vec[0][0] * vec[1][1] - vec[0][1] * vec[1][0]) * factor;
}


void write_new_xml(const std::vector<FcsClassExtent> fc2_in, const std::string xml_out)
{
    // Write to XML file

    int i, j;
    using boost::property_tree::ptree;

    ptree pt;
    string str_pos[3], str_tmp;

    str_tmp.clear();

    pt.put("Data.OriginalFC2", original_xml);
    pt.put("Data.SCPH_file", file_fc2_correction);
    pt.put("Data.SCPH_Temperature", double2string(temp, 5));
    pt.put("Data.Structure.NumberOfAtoms", nat);
    pt.put("Data.Structure.NumberOfElements", nkd);

    for (i = 0; i < nkd; ++i) {
        ptree &child = pt.add("Data.Structure.AtomicElements.element", kd_symbol[i]);
        child.put("<xmlattr>.number", i + 1);
    }

    for (i = 0; i < 3; ++i) {
        str_pos[i].clear();
        for (j = 0; j < 3; ++j) {
            str_pos[i] += " " + double2string(lavec_s[j][i]);
        }
    }
    pt.put("Data.Structure.LatticeVector", "");
    pt.put("Data.Structure.LatticeVector.a1", str_pos[0]);
    pt.put("Data.Structure.LatticeVector.a2", str_pos[1]);
    pt.put("Data.Structure.LatticeVector.a3", str_pos[2]);

    std::stringstream ss;
    ss << is_periodic[0] << " " << is_periodic[1] << " " << is_periodic[2];
    pt.put("Data.Structure.Periodicity", ss.str());

    pt.put("Data.Structure.Position", "");

    for (i = 0; i < nat; ++i) {
        str_tmp.clear();
        for (j = 0; j < 3; ++j) str_tmp += " " + double2string(xr_s[i][j]);
        ptree &child = pt.add("Data.Structure.Position.pos", str_tmp);
        child.put("<xmlattr>.index", i + 1);
        child.put("<xmlattr>.element", kd_symbol[kd[i]]);
    }

    pt.put("Data.Symmetry.NumberOfTranslations", ntran);
    for (i = 0; i < ntran; ++i) {
        for (j = 0; j < natmin; ++j) {
            ptree &child = pt.add("Data.Symmetry.Translations.map", map_p2s[j][i] + 1);
            child.put("<xmlattr>.tran", i + 1);
            child.put("<xmlattr>.atom", j + 1);
        }
    }


    pt.put("Data.ForceConstants", "");
    str_tmp.clear();

    for (const auto &it: fc2_in) {
        ptree &child = pt.add("Data.ForceConstants.HARMONIC.FC2", double2string(it.fcs_val));

        child.put("<xmlattr>.pair1", std::to_string(it.atm1 + 1) + " " + std::to_string(it.xyz1 + 1));
        child.put("<xmlattr>.pair2",
                  std::to_string(it.atm2 + 1) + " " + std::to_string(it.xyz2 + 1) + " " +
                      std::to_string(it.cell_s + 1));
    }

    using namespace boost::property_tree::xml_parser;
    const int indent = 2;

#if BOOST_VERSION >= 105600
    write_xml(xml_out,
              pt,
              std::locale(),
              xml_writer_make_settings<ptree::key_type>(' ', indent, widen<std::string>("utf-8")));
#else
    write_xml(xml_out, pt, std::locale(), xml_writer_make_settings(' ', indent, widen<char>("utf-8")));
#endif
}


string double2string(const double d, const int nprec)
{
    std::string rt;
    std::stringstream ss;

    ss << std::scientific << std::setprecision(nprec) << d;
    ss >> rt;
    return rt;
}
