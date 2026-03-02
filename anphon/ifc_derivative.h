#pragma once

#include <Eigen/Core>
#include <complex>
#include <string>
#include <utility>
#include <vector>
#include "fcs_phonon.h"
#include "pointers.h"

namespace PHON_NS
{

class DerivativeIFC: protected Pointers
{
public:
    explicit DerivativeIFC(class PHON *phon);
    ~DerivativeIFC() = default;

    bool check_del_v_strain_in_real_space_equivalence(const std::vector<FcsArrayWithCell> &fcs_aligned,
                                                      const std::vector<std::pair<int, int>> &strain_components) const;

    bool check_del_v_strain_in_real_space_equivalence_verbose(
        const std::vector<FcsArrayWithCell> &fcs_aligned,
        const std::vector<std::pair<int, int>> &strain_components,
        std::string &mismatch_message) const;

    void compute_del_v_strain_in_real_space(const std::vector<FcsArrayWithCell> &fcs_aligned,
                                            std::vector<FcsArrayWithCell> &delta_fcs,
                                            const std::vector<std::pair<int, int>> &strain_components) const;

    void compute_del_v_strain_in_real_space1(const std::vector<FcsArrayWithCell> &fcs_aligned,
                                             std::vector<FcsArrayWithCell> &delta_fcs,
                                             int ixyz1,
                                             int ixyz2) const;

    void compute_del_v_strain_in_real_space2(const std::vector<FcsArrayWithCell> &fcs_aligned,
                                             std::vector<FcsArrayWithCell> &delta_fcs,
                                             int ixyz11,
                                             int ixyz12,
                                             int ixyz21,
                                             int ixyz22) const;

    void compute_del_v_strain_in_real_space1_legacy(const std::vector<FcsArrayWithCell> &fcs_aligned,
                                                    std::vector<FcsArrayWithCell> &delta_fcs,
                                                    int ixyz1,
                                                    int ixyz2) const;

    void compute_del_v_strain_in_real_space2_legacy(const std::vector<FcsArrayWithCell> &fcs_aligned,
                                                    std::vector<FcsArrayWithCell> &delta_fcs,
                                                    int ixyz11,
                                                    int ixyz12,
                                                    int ixyz21,
                                                    int ixyz22) const;
};

} // namespace PHON_NS
