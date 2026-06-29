// _alm.cpp -- nanobind binding for the ALAMODE 2.0dev ALM C++ class.
//
// Fresh binding (no legacy C wrapper / Cython): nanobind holds the ALM_NS::ALM object directly.
// Chosen over pybind11 for smaller/faster output and STABLE_ABI (abi3) wheels. Raw double*/int*
// C++ methods are wrapped with lambdas that size outputs from ALM's own count getters and return
// capsule-owned NumPy arrays.
//
// Build: see python/CMakeLists.txt (nanobind_add_module + the 2.0dev alm sources).

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "alm.h"
#include "optimize.h"

namespace nb = nanobind;
using ALM_NS::ALM;
using ALM_NS::OptimizerControl;

// Contiguous CPU input views (the Python layer guarantees dtype/order, so require exact types).
using f64in = nb::ndarray<const double, nb::c_contig, nb::device::cpu>;
using i32in = nb::ndarray<const int, nb::c_contig, nb::device::cpu>;

// Allocate a 1-D NumPy array of length n, owned by a capsule; hand back the raw pointer to fill.
template <typename T>
static nb::ndarray<nb::numpy, T> make_owned(std::size_t n, T **out)
{
    T *data = new T[n]();
    *out = data;
    nb::capsule owner(data, [](void *p) noexcept { delete[] static_cast<T *>(p); });
    return nb::ndarray<nb::numpy, T>(data, {n}, owner);
}

// Total number of irreducible FC parameters over all orders (= ncols of the sensing matrix).
static std::size_t total_irred(ALM &self)
{
    const int maxorder = self.get_maxorder();
    std::size_t n = 0;
    for (int o = 1; o <= maxorder; ++o) n += self.get_number_of_irred_fc_elements(o);
    return n;
}

NB_MODULE(_alm, m)
{
    m.doc() = "nanobind binding for the ALAMODE 2.0dev ALM library";

    // ---------------- OptimizerControl ----------------
    nb::class_<OptimizerControl>(m, "OptimizerControl")
        .def(nb::init<>())
        .def_rw("linear_model", &OptimizerControl::linear_model)
        .def_rw("use_sparse_solver", &OptimizerControl::use_sparse_solver)
        .def_rw("sparsesolver", &OptimizerControl::sparsesolver)
        .def_rw("use_cholesky", &OptimizerControl::use_cholesky)
        .def_rw("chunk_size", &OptimizerControl::chunk_size)
        .def_rw("maxnum_iteration", &OptimizerControl::maxnum_iteration)
        .def_rw("tolerance_iteration", &OptimizerControl::tolerance_iteration)
        .def_rw("output_frequency", &OptimizerControl::output_frequency)
        .def_rw("standardize", &OptimizerControl::standardize)
        .def_rw("displacement_normalization_factor",
                &OptimizerControl::displacement_normalization_factor)
        .def_rw("debiase_after_l1opt", &OptimizerControl::debiase_after_l1opt)
        .def_rw("l1_solver", &OptimizerControl::l1_solver)
        .def_rw("efit_weight", &OptimizerControl::efit_weight)
        .def_rw("efit_escale", &OptimizerControl::efit_escale)
        .def_rw("efit_cv", &OptimizerControl::efit_cv)
        .def_rw("cross_validation", &OptimizerControl::cross_validation)
        .def_rw("l1_alpha", &OptimizerControl::l1_alpha)
        .def_rw("l1_alpha_min", &OptimizerControl::l1_alpha_min)
        .def_rw("l1_alpha_max", &OptimizerControl::l1_alpha_max)
        .def_rw("l1_alpha_min_ratio", &OptimizerControl::l1_alpha_min_ratio)
        .def_rw("num_l1_alpha", &OptimizerControl::num_l1_alpha)
        .def_rw("l1_ratio", &OptimizerControl::l1_ratio)
        .def_rw("save_solution_path", &OptimizerControl::save_solution_path)
        .def_rw("stop_criterion", &OptimizerControl::stop_criterion)
        .def_rw("periodic_image_conv", &OptimizerControl::periodic_image_conv);

    // ---------------- ALM ----------------
    nb::class_<ALM>(m, "ALMCore")
        .def(nb::init<>())

        // -- verbosity / output ------------------------------------------------
        .def("set_verbosity", &ALM::set_verbosity)
        .def("get_verbosity", &ALM::get_verbosity)
        .def("set_output_filename_prefix", &ALM::set_output_filename_prefix)
        .def("set_print_symmetry", &ALM::set_print_symmetry)
        .def("set_symmetry_tolerance", &ALM::set_symmetry_tolerance)
        .def("get_symmetry_tolerance", &ALM::get_symmetry_tolerance)

        // -- cell --------------------------------------------------------------
        .def("set_cell",
             [](ALM &self, f64in lavec, f64in xcoord, i32in kind) {
                 if (lavec.ndim() != 2 || lavec.shape(0) != 3 || lavec.shape(1) != 3)
                     throw std::invalid_argument("set_cell: lavec must have shape (3, 3)");
                 if (xcoord.ndim() != 2 || xcoord.shape(1) != 3)
                     throw std::invalid_argument("set_cell: xcoord must have shape (nat, 3)");
                 if (kind.ndim() != 1 || kind.shape(0) != xcoord.shape(0))
                     throw std::invalid_argument("set_cell: kind must have shape (nat,)");
                 auto nat = static_cast<std::size_t>(xcoord.shape(0));
                 const auto *lv = reinterpret_cast<const double(*)[3]>(lavec.data());
                 const auto *xc = reinterpret_cast<const double(*)[3]>(xcoord.data());
                 self.set_cell(nat, lv, xc, kind.data());
             },
             nb::arg("lavec").noconvert(), nb::arg("xcoord").noconvert(), nb::arg("kind").noconvert())
        .def("set_element_names", &ALM::set_element_names)
        .def("set_periodicity",
             [](ALM &self, i32in is_periodic) {
                 if (is_periodic.ndim() != 1 || is_periodic.shape(0) != 3)
                     throw std::invalid_argument("set_periodicity: expected shape (3,)");
                 int p[3] = {is_periodic.data()[0], is_periodic.data()[1], is_periodic.data()[2]};
                 self.set_periodicity(p);
             },
             nb::arg("is_periodic").noconvert())

        // -- training / validation data ---------------------------------------
        .def("set_u_train",
             [](ALM &self, f64in u) {
                 if (u.ndim() != 2) throw std::invalid_argument("set_u_train: u must be 2-D (nsnap, 3*nat)");
                 const auto nrow = static_cast<std::size_t>(u.shape(0));
                 const auto ncol = static_cast<std::size_t>(u.shape(1));
                 const double *d = u.data();
                 std::vector<std::vector<double>> uv(nrow, std::vector<double>(ncol));
                 for (std::size_t i = 0; i < nrow; ++i)
                     for (std::size_t j = 0; j < ncol; ++j) uv[i][j] = d[i * ncol + j];
                 self.set_u_train(uv);
             },
             nb::arg("u").noconvert())
        .def("set_f_train",
             [](ALM &self, f64in f) {
                 if (f.ndim() != 2) throw std::invalid_argument("set_f_train: f must be 2-D (nsnap, 3*nat)");
                 const auto nrow = static_cast<std::size_t>(f.shape(0));
                 const auto ncol = static_cast<std::size_t>(f.shape(1));
                 const double *d = f.data();
                 std::vector<std::vector<double>> fv(nrow, std::vector<double>(ncol));
                 for (std::size_t i = 0; i < nrow; ++i)
                     for (std::size_t j = 0; j < ncol; ++j) fv[i][j] = d[i * ncol + j];
                 self.set_f_train(fv);
             },
             nb::arg("f").noconvert())
        .def("set_validation_data",
             [](ALM &self, f64in u, f64in f) {
                 auto to_vv = [](const f64in &a) {
                     if (a.ndim() != 2) throw std::invalid_argument("set_validation_data: u/f must be 2-D");
                     const auto nr = static_cast<std::size_t>(a.shape(0));
                     const auto nc = static_cast<std::size_t>(a.shape(1));
                     const double *d = a.data();
                     std::vector<std::vector<double>> v(nr, std::vector<double>(nc));
                     for (std::size_t i = 0; i < nr; ++i)
                         for (std::size_t j = 0; j < nc; ++j) v[i][j] = d[i * nc + j];
                     return v;
                 };
                 self.set_validation_data(to_vv(u), to_vv(f));
             },
             nb::arg("u").noconvert(), nb::arg("f").noconvert())
        .def("get_u_train", &ALM::get_u_train)
        .def("get_f_train", &ALM::get_f_train)
        .def("get_number_of_data", &ALM::get_number_of_data)
        .def("get_nrows_sensing_matrix", &ALM::get_nrows_sensing_matrix)

        // -- model definition --------------------------------------------------
        .def("define",
             [](ALM &self, int maxorder, std::size_t nkd, i32in nbody_include, f64in cutoff_radii) {
                 const double *cut = (cutoff_radii.size() > 0) ? cutoff_radii.data() : nullptr;
                 self.define(maxorder, nkd, nbody_include.data(), cut);
             },
             nb::arg("maxorder"), nb::arg("nkd"), nb::arg("nbody_include").noconvert(),
             nb::arg("cutoff_radii").noconvert())
        .def("set_forceconstant_basis", &ALM::set_forceconstant_basis)
        .def("get_forceconstant_basis", &ALM::get_forceconstant_basis)
        .def("init_fc_table", &ALM::init_fc_table)
        .def("ready_all_constraints", &ALM::ready_all_constraints)
        .def("get_maxorder", &ALM::get_maxorder)

        // -- constraints / fixing ---------------------------------------------
        .def("set_constraint_mode", &ALM::set_constraint_mode)
        .def("set_algebraic_constraint", &ALM::set_algebraic_constraint)
        .def("set_rotation_axis", &ALM::set_rotation_axis)
        .def("set_fc_file", &ALM::set_fc_file)  // e.g. FC2FIX: set_fc_file(2, "supercell.h5")
        .def("set_fc_fix", &ALM::set_fc_fix)
        .def("set_forceconstants_to_fix",
             [](ALM &self, const std::vector<std::vector<int>> &intpair_fix,
                const std::vector<double> &values_fix) {
                 if (intpair_fix.empty())
                     throw std::invalid_argument("set_forceconstants_to_fix: intpair_fix is empty");
                 if (values_fix.size() != intpair_fix.size())
                     throw std::invalid_argument("set_forceconstants_to_fix: values_fix and "
                                                 "intpair_fix must have the same length");
                 for (const auto &row : intpair_fix)
                     if (row.size() < 2)
                         throw std::invalid_argument("set_forceconstants_to_fix: each intpair row "
                                                     "must have >= 2 flattened indices");
                 self.set_forceconstants_to_fix(intpair_fix, values_fix);
             })

        // -- optimizer control -------------------------------------------------
        .def("set_optimizer_control", &ALM::set_optimizer_control)
        .def("get_optimizer_control", &ALM::get_optimizer_control)
        .def("get_cv_l1_alpha", &ALM::get_cv_l1_alpha)

        // -- run (release the GIL: long, OpenMP-parallel, no Python callbacks) -----------------
        .def("run_optimize", &ALM::run_optimize, nb::call_guard<nb::gil_scoped_release>())
        .def("run_suggest", &ALM::run_suggest, nb::call_guard<nb::gil_scoped_release>())

        // -- generic input dict ------------------------------------------------
        .def("set_input_vars",
             [](ALM &self, const std::map<std::string, std::string> &d) { self.set_input_vars(d); })
        .def("get_input_var", &ALM::get_input_var)

        // -- FC element counts -------------------------------------------------
        .def("get_number_of_fc_elements", &ALM::get_number_of_fc_elements)
        .def("get_number_of_irred_fc_elements", &ALM::get_number_of_irred_fc_elements)
        .def("get_number_of_fc_origin", &ALM::get_number_of_fc_origin)
        .def("get_atom_mapping_by_pure_translations", &ALM::get_atom_mapping_by_pure_translations)

        // -- sensing matrix A, b ----------------------------------------------
        .def("get_matrix_elements",
             [](ALM &self) {
                 if (self.get_number_of_data() == 0)
                     throw std::runtime_error("get_matrix_elements: training data not set "
                                              "(call set_u_train/set_f_train first)");
                 const std::size_t nrows = self.get_nrows_sensing_matrix();
                 const std::size_t ncols = total_irred(self);
                 double *ap = nullptr, *bp = nullptr;
                 auto amat = make_owned<double>(nrows * ncols, &ap);
                 auto bvec = make_owned<double>(nrows, &bp);
                 {
                     // Heavy, OpenMP-parallel build into raw buffers: drop the GIL.
                     nb::gil_scoped_release rel;
                     self.get_matrix_elements(ap, bp);
                 }
                 return nb::make_tuple(amat, bvec, nrows, ncols);
             })

        // -- get / set FCs -----------------------------------------------------
        .def("get_fc_origin",
             [](ALM &self, int fc_order, int permutation) {
                 const std::size_t n = self.get_number_of_fc_origin(fc_order, permutation);
                 double *vp = nullptr; int *ip = nullptr;
                 auto vals = make_owned<double>(n, &vp);
                 auto idx = make_owned<int>(n * (fc_order + 1), &ip);
                 self.get_fc_origin(vp, ip, fc_order, permutation);
                 return nb::make_tuple(vals, idx);
             },
             nb::arg("fc_order"), nb::arg("permutation") = 1)
        .def("get_fc_irreducible",
             [](ALM &self, int fc_order) {
                 const std::size_t n = self.get_number_of_irred_fc_elements(fc_order);
                 double *vp = nullptr; int *ip = nullptr;
                 auto vals = make_owned<double>(n, &vp);
                 auto idx = make_owned<int>(n * (fc_order + 1), &ip);
                 self.get_fc_irreducible(vp, ip, fc_order);
                 return nb::make_tuple(vals, idx);
             },
             nb::arg("fc_order"))
        .def("get_fc_all",
             [](ALM &self, int fc_order, int permutation) {
                 // map_p2s has shape (nat_primitive, ntran); get_fc_all replicates the origin set
                 // over the ntran pure translations -> use the inner dimension.
                 const auto &map_p2s = self.get_atom_mapping_by_pure_translations();
                 const std::size_t ntrans =
                     (map_p2s.empty() || map_p2s[0].empty()) ? 0 : map_p2s[0].size();
                 const std::size_t n = self.get_number_of_fc_origin(fc_order, permutation) * ntrans;
                 double *vp = nullptr; int *ip = nullptr;
                 auto vals = make_owned<double>(n, &vp);
                 auto idx = make_owned<int>(n * (fc_order + 1), &ip);
                 self.get_fc_all(vp, ip, fc_order, permutation);
                 return nb::make_tuple(vals, idx);
             },
             nb::arg("fc_order"), nb::arg("permutation") = 1)
        .def("set_fc",
             [](ALM &self, f64in fc_in) {
                 const std::size_t n = total_irred(self);
                 if (static_cast<std::size_t>(fc_in.size()) != n)
                     throw std::invalid_argument("set_fc: expected " + std::to_string(n) +
                                                 " irreducible FCs, got " + std::to_string(fc_in.size()));
                 std::vector<double> buf(fc_in.data(), fc_in.data() + fc_in.size());
                 self.set_fc(buf.data());
             },
             nb::arg("fc_in").noconvert())
        .def("set_fc_zero_threshold", &ALM::set_fc_zero_threshold)

        // -- displacement patterns (suggest mode) -----------------------------
        .def("get_number_of_displacement_patterns", &ALM::get_number_of_displacement_patterns)
        .def("get_number_of_displaced_atoms",
             [](ALM &self, int fc_order) {
                 const std::size_t npat = self.get_number_of_displacement_patterns(fc_order);
                 int *np_ = nullptr;
                 auto nums = make_owned<int>(npat, &np_);
                 self.get_number_of_displaced_atoms(np_, fc_order);
                 return nums;
             },
             nb::arg("fc_order"))
        .def("get_displacement_patterns",
             [](ALM &self, int fc_order) {
                 const std::size_t npat = self.get_number_of_displacement_patterns(fc_order);
                 std::vector<int> nums(npat);
                 self.get_number_of_displaced_atoms(nums.data(), fc_order);
                 std::size_t tot = 0;
                 for (auto v : nums) tot += static_cast<std::size_t>(v);
                 int *ai = nullptr; double *dp = nullptr;
                 auto atom_indices = make_owned<int>(tot, &ai);
                 auto disp = make_owned<double>(tot * 3, &dp);
                 const int nbasis = self.get_displacement_patterns(ai, dp, fc_order);
                 return nb::make_tuple(atom_indices, disp, nbasis);
             },
             nb::arg("fc_order"))

        // -- save --------------------------------------------------------------
        .def("set_fcs_save_flag", &ALM::set_fcs_save_flag)
        .def("save_fc",
             [](ALM &self, const std::string &filename, const std::string &fc_format,
                int maxorder_to_save) {
                 if (maxorder_to_save < 0) maxorder_to_save = self.get_maxorder();
                 nb::gil_scoped_release rel;
                 self.save_fc(filename, fc_format, maxorder_to_save);
             },
             nb::arg("filename"), nb::arg("fc_format") = "alamode", nb::arg("maxorder_to_save") = -1);
}
