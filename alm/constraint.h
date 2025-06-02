/*
 constraint.h

 Copyright (c) 2014, 2015, 2016 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <unordered_map>
#include <boost/bimap.hpp>
#include <utility>
#include <vector>
#include <string>
#include <iomanip>
#include <map>
#include <unordered_set>
#include "constants.h"
#include "fcs.h"
#include "cluster.h"
#include "system.h"
#include "timer.h"


namespace ALM_NS
{
class ConstraintClass
{
public:
    std::vector<double> w_const;

    ConstraintClass() = default;

    ConstraintClass(const ConstraintClass &a) = default;

    ConstraintClass(std::vector<double> vec) :
        w_const(std::move(vec)) {}

    ConstraintClass(const int n,
                    const double *arr,
                    const int nshift = 0)
    {
        for (auto i = nshift; i < n; ++i) {
            w_const.push_back(arr[i]);
        }
    }

    bool operator<(const ConstraintClass &a) const
    {
        return std::lexicographical_compare(w_const.begin(),
                                            w_const.end(),
                                            a.w_const.begin(),
                                            a.w_const.end());
    }
};

class ConstraintTypeFix
{
public:
    size_t p_index_target;
    double val_to_fix;

    ConstraintTypeFix(const size_t index_in,
                      const double val_in) :
        p_index_target(index_in), val_to_fix(val_in) {}
};

class ConstraintTypeRelate
{
public:
    size_t p_index_target;
    std::vector<double> alpha;
    std::vector<size_t> p_index_orig;

    ConstraintTypeRelate(const size_t index_in,
                         std::vector<double> alpha_in,
                         std::vector<size_t> p_index_in) :
        p_index_target(index_in), alpha(std::move(alpha_in)),
        p_index_orig(std::move(p_index_in)) {}
};

inline bool equal_within_eps12(const std::vector<double> &a,
                               const std::vector<double> &b)
{
    const auto n = a.size();
    const auto m = b.size();
    if (n != m) return false;
    for (size_t i = 0; i < n; ++i) {
        if (std::abs(a[i] - b[i]) > eps12) return false;
    }
    return true;
}

class ConstraintIntegerElement
{
    // For sparse representation
public:
    size_t col;
    int val;

    ConstraintIntegerElement(const size_t col_in,
                             const int val_in) :
        col(col_in), val(val_in) {}
};

// Operator for sort
inline bool operator<(const std::vector<ConstraintIntegerElement> &obj1,
                      const std::vector<ConstraintIntegerElement> &obj2)
{
    const auto len1 = obj1.size();
    const auto len2 = obj2.size();
    const auto min = (std::min)(len1, len2);

    for (size_t i = 0; i < min; ++i) {
        if (obj1[i].col < obj2[i].col) {
            return true;
        }
        if (obj1[i].col > obj2[i].col) {
            return false;
        }
        if (obj1[i].val < obj2[i].val) {
            return true;
        }
        if (obj1[i].val > obj2[i].val) {
            return false;
        }
    }
    return false;
}

// Operator for unique
inline bool operator==(const std::vector<ConstraintIntegerElement> &obj1,
                       const std::vector<ConstraintIntegerElement> &obj2)
{
    const auto len1 = obj1.size();
    const auto len2 = obj2.size();
    if (len1 != len2) return false;

    for (size_t i = 0; i < len1; ++i) {
        if (obj1[i].col != obj2[i].col || obj1[i].val != obj2[i].val) {
            return false;
        }
    }
    return true;
}

class ConstraintDoubleElement
{
    // For sparse representation
public:
    size_t col;
    double val;

    ConstraintDoubleElement(const size_t col_in,
                            const double val_in) :
        col(col_in), val(val_in) {}

    bool operator<(const ConstraintDoubleElement &obj) const
    {
        return col < obj.col;
    }
};

// Operator for sort
inline bool operator<(const std::vector<ConstraintDoubleElement> &obj1,
                      const std::vector<ConstraintDoubleElement> &obj2)
{
    const auto len1 = obj1.size();
    const auto len2 = obj2.size();
    const auto min = (std::min)(len1, len2);

    for (size_t i = 0; i < min; ++i) {
        if (obj1[i].col < obj2[i].col) {
            return true;
        }
        if (obj1[i].col > obj2[i].col) {
            return false;
        }
        if (obj1[i].val < obj2[i].val) {
            return true;
        }
        if (obj1[i].val > obj2[i].val) {
            return false;
        }
    }
    return false;
}

// Operator for unique
inline bool operator==(const std::vector<ConstraintDoubleElement> &obj1,
                       const std::vector<ConstraintDoubleElement> &obj2)
{
    const auto len1 = obj1.size();
    const auto len2 = obj2.size();
    if (len1 != len2) return false;

    for (size_t i = 0; i < len1; ++i) {
        if (obj1[i].col != obj2[i].col || (std::abs(obj1[i].val - obj2[i].val) > 1.0e-10)) {
            return false;
        }
    }
    return true;
}

inline bool operator<(const std::map<size_t, double> &obj1,
                      const std::map<size_t, double> &obj2)
{
    return obj1.begin()->first < obj2.begin()->first;
}

class Constraint
{
public:
    Constraint();

    ~Constraint();

    void setup(const std::unique_ptr<System> &system,
               const std::unique_ptr<Fcs> &fcs,
               const std::unique_ptr<Cluster> &cluster,
               const std::unique_ptr<Symmetry> &symmetry,
               const int linear_model,
               const int periodic_image_conv,
               const int verbosity,
               std::unique_ptr<Timer> &timer);

    void get_mapping_constraint(const int nmax,
                                const std::vector<size_t> *nequiv,
                                const ConstraintSparseForm *const_in,
                                std::vector<ConstraintTypeFix> *const_fix_out,
                                std::vector<ConstraintTypeRelate> *const_relate_out,
                                boost::bimap<size_t, size_t> *index_bimap_out) const;

    [[nodiscard]] int get_constraint_mode() const;

    void set_constraint_mode(const int);

    [[nodiscard]] size_t get_number_of_constraints() const;

    [[nodiscard]] std::string get_fc_file(const int) const;

    void set_fc_file(const int,
                     const std::string);

    [[nodiscard]] bool get_fix_harmonic() const;

    void set_fix_harmonic(const bool);

    bool get_fix_cubic() const;

    void set_fix_cubic(const bool);

    void set_constraint_algebraic(const int constraint_algebraic_in);

    [[nodiscard]] int get_constraint_algebraic() const;

    [[nodiscard]] double **get_const_mat() const;

    [[nodiscard]] double *get_const_rhs() const;

    [[nodiscard]] double get_tolerance_constraint() const;

    void set_tolerance_constraint(const double);

    [[nodiscard]] bool get_exist_constraint() const;

    [[nodiscard]] std::string get_rotation_axis() const;

    void set_rotation_axis(const std::string);

    [[nodiscard]] const ConstraintSparseForm &get_const_symmetry(const int) const;

    [[nodiscard]] const std::vector<ConstraintTypeFix> &get_const_fix(const int) const;

    void set_const_fix_val_to_fix(const int order,
                                  const size_t idx,
                                  const double val);

    [[nodiscard]] const std::vector<ConstraintTypeRelate> &get_const_relate(const int) const;

    [[nodiscard]] const boost::bimap<size_t, size_t> &get_index_bimap(const int) const;

    void set_constraint_flag(const std::string const_name, const int use_constraint);

    void show_status_constraint() const;

    void set_forceconstants_to_fix(const std::vector<std::vector<int>> &intpair_fix,
                                   const std::vector<double> &values_fix);

    void update_constraint_matrix(const std::unique_ptr<System> &system,
                                  const std::unique_ptr<Symmetry> &symmetry,
                                  const std::unique_ptr<Cluster> &cluster,
                                  const std::unique_ptr<Fcs> &fcs,
                                  const int verbosity,
                                  const int periodic_image_conv);

    [[nodiscard]] bool ready_all_constraints() const;

private:
    int constraint_mode;
    size_t number_of_constraints;
    std::string fc2_file, fc3_file;
    bool fix_harmonic, fix_cubic;
    int constraint_algebraic;

    double **const_mat;
    double *const_rhs;

    double tolerance_constraint;

    std::string rotation_axis;
    std::vector<std::vector<ConstraintTypeFix>> const_fix;
    std::vector<std::vector<ConstraintTypeRelate>> const_relate;
    std::vector<std::vector<ConstraintTypeRelate>> const_relate_rotation;
    boost::bimap<size_t, size_t> *index_bimap;

    bool impose_inv_T, impose_inv_R, exclude_last_R, impose_inv_Huang;

    std::vector<ConstraintSparseForm> const_symmetry;
    std::vector<ConstraintSparseForm> const_translation;
    std::vector<ConstraintSparseForm> const_rotation_self;
    std::vector<ConstraintSparseForm> const_rotation_cross;
    std::vector<ConstraintSparseForm> const_huang;
    std::vector<ConstraintSparseForm> const_self;

    std::vector<std::vector<int>> intpair_fix_fc2, intpair_fix_fc3;
    std::vector<double> values_fix_fc2, values_fix_fc3;


    std::map<std::string, int> status_constraint_subset; // -1: not used,
    //  0: used but not ready,
    //  1: ready


    void set_default_variables();

    void deallocate_variables();

    [[nodiscard]] int levi_civita(const int,
                                  const int,
                                  const int) const;

    void generate_rotational_constraint(const std::unique_ptr<System> &system,
                                        const std::unique_ptr<Symmetry> &symmetry,
                                        const std::unique_ptr<Cluster> &cluster,
                                        const std::unique_ptr<Fcs> &fcs,
                                        const int,
                                        const double);


    // const_mat and const_rhs are updated.
    [[nodiscard]] size_t calc_constraint_matrix(const int maxorder,
                                                const std::vector<size_t> *nequiv,
                                                const size_t nparams) const;

    void print_constraint(const ConstraintSparseForm &) const;

    void print_constraint_information(const std::unique_ptr<Cluster> &cluster) const;

    void setup_rotation_axis(bool [3][3]);

    [[nodiscard]] bool is_allzero(const int,
                                  const double *,
                                  const int nshift = 0) const;

    [[nodiscard]] bool is_allzero(const std::vector<int> &,
                                  int &) const;

    [[nodiscard]] bool is_allzero(const std::vector<double> &,
                                  const double,
                                  int &,
                                  const int nshift = 0) const;


    void remove_redundant_rows(const size_t n,
                               std::vector<ConstraintClass> &Constraint_vec,
                               const double tolerance = eps12) const;

    // const_symmetry is updated.
    void generate_symmetry_constraint(const size_t nat,
                                      const std::unique_ptr<Symmetry> &symmetry,
                                      const std::unique_ptr<Cluster> &cluster,
                                      const std::unique_ptr<Fcs> &fcs,
                                      const int verbosity);

    void generate_fix_constraint(const std::unique_ptr<Symmetry> &symmetry,
                                 const std::unique_ptr<Fcs> &fcs);

    void get_constraint_translation(const Cell &supercell,
                                    const std::unique_ptr<Symmetry> &symmetry,
                                    const std::unique_ptr<Cluster> &cluster,
                                    const std::unique_ptr<Fcs> &fcs,
                                    const int order,
                                    const std::vector<FcProperty> &fc_table,
                                    const size_t nparams,
                                    ConstraintSparseForm &const_out,
                                    const bool do_rref = false) const;

    void get_constraint_translation_for_periodic_images(const Cell &supercell,
                                                        const std::unique_ptr<Symmetry> &symmetry,
                                                        const std::unique_ptr<Cluster> &cluster,
                                                        const std::unique_ptr<Fcs> &fcs,
                                                        const int order,
                                                        const std::vector<FcProperty> &fc_table,
                                                        const size_t nparams,
                                                        ConstraintSparseForm &const_out,
                                                        const bool do_rref) const;

    // const_translation is updated.
    void generate_translational_constraint(const Cell &,
                                           const std::unique_ptr<Symmetry> &symmetry,
                                           const std::unique_ptr<Cluster> &cluster,
                                           const std::unique_ptr<Fcs> &fcs,
                                           const int,
                                           const int);

    void generate_huang_constraint(const Cell &,
                                   const std::unique_ptr<Symmetry> &symmetry,
                                   const std::unique_ptr<Cluster> &cluster,
                                   const std::unique_ptr<Fcs> &fcs,
                                   const std::vector<Eigen::MatrixXd> &x_image,
                                   const int verbosity);


    void get_forceconstants_from_file(const int order,
                                      const std::unique_ptr<Symmetry> &symmetry,
                                      const std::unique_ptr<Fcs> &fcs,
                                      const std::string file_to_fix,
                                      std::vector<std::vector<int>> &intpair_fcs,
                                      std::vector<double> &fcs_values) const;

    static void parse_forceconstants_from_xml(const int order,
                                              const std::unique_ptr<Symmetry> &symmetry,
                                              const std::unique_ptr<Fcs> &fcs,
                                              const std::string file_to_fix,
                                              std::vector<std::vector<int>> &intpair_fcs,
                                              std::vector<double> &fcs_values);

    static void parse_forceconstants_from_h5(const int order,
                                             const std::unique_ptr<Symmetry> &symmetry,
                                             const std::unique_ptr<Fcs> &fcs,
                                             const std::string file_to_fix,
                                             std::vector<std::vector<int>> &intpair_fcs,
                                             std::vector<double> &fcs_values);

    void set_rotation_constraints(const std::unique_ptr<System> &system,
                                  const std::unique_ptr<Symmetry> &symmetry,
                                  const std::unique_ptr<Cluster> &cluster,
                                  const std::unique_ptr<Fcs> &fcs,
                                  const int order,
                                  const bool valid_rotation_axis[3][3],
                                  const std::unordered_set<FcProperty> &list_found,
                                  const std::unordered_set<FcProperty> &list_found_last,
                                  const double tolerance,
                                  std::vector<std::vector<ConstraintDoubleElement>> *const_self_vec,
                                  std::vector<std::vector<ConstraintDoubleElement>> *const_cross_vec);

    void set_rotation_constraints_extra(const std::unique_ptr<System> &system,
                                        const std::unique_ptr<Symmetry> &symmetry,
                                        const std::unique_ptr<Cluster> &cluster,
                                        const std::unique_ptr<Fcs> &fcs,
                                        const int order,
                                        const bool valid_rotation_axis[3][3],
                                        const std::unordered_set<FcProperty> &list_found,
                                        const double tolerance,
                                        std::vector<std::vector<ConstraintDoubleElement>> *const_self_vec,
                                        std::vector<std::vector<ConstraintDoubleElement>> *const_cross_vec);

    void test_svd(ConstraintSparseForm &const_in, const int nparams) const;
};

extern "C"
{
void dgesvd_(char *jobu,
             char *jobvt,
             int *m,
             int *n,
             double *a,
             int *lda,
             double *s,
             double *u,
             int *ldu,
             double *vt,
             int *ldvt,
             double *work,
             int *lwork,
             int *info);
}
}
