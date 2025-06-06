extern "C"
{
#include "../external/niggli.h"
}

#include <Eigen/Core>
#include <Eigen/LU>

inline auto niggli_reduction(const Eigen::Matrix3d &lat_in, Eigen::Matrix3d &lat_out, Eigen::Matrix3d &c_matrix,
                             const double eps = 1.0e-8) -> int
{
    // lat_in: 3x3 matrix in the form of (a_x, b_x, c_x)
    //                                   (a_y, b_y, c_y)
    //                                   (a_z, b_z, c_z)
    //         This input matrix may need to be transposed beforehand
    //         when lat_in is reciprocal lattice vectors.
    //
    // lat_out: 3x3 matrix storing the Niggli reduced cell
    // c_matrix: lat_out = lat_in * c_matrix

    double lat[9];

    int k = 0;
    for (auto i = 0; i < 3; ++i) {
        for (auto j = 0; j < 3; ++j) {
            lat[k++] = lat_in(i, j);
        }
    }

    int succeeded = niggli_reduce(lat, eps);

    k = 0;
    for (auto i = 0; i < 3; ++i) {
        for (auto j = 0; j < 3; ++j) {
            lat_out(i, j) = lat[k++];
        }
    }

    c_matrix = lat_in.inverse() * lat_out;

    return succeeded;
}
