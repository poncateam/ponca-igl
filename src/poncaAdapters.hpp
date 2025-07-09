/*
This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

/// \file This file contains structures mapping IGL data representation to Ponca concepts
/// \author Nicolas Mellado <nmellado0@gmail.com>

#include <Eigen/Core>


/// Map a block to a Ponca point
template <typename _Scalar>
struct BlockPointAdapter {
public:
    enum {Dim = 3};
    using Scalar     = _Scalar;
    using VectorType = Eigen::Matrix<Scalar, Dim, 1>;
    using MatrixType = Eigen::Matrix<Scalar, Dim, Dim>;

    using InternalType = typename Eigen::Block<const Eigen::MatrixXd, 1, Eigen::Dynamic>::ConstTransposeReturnType;

    /// \brief Map a vector as ponca Point
    PONCA_MULTIARCH inline BlockPointAdapter(InternalType v, InternalType n)
            : m_pos (v), m_nor (n) {}

    PONCA_MULTIARCH inline InternalType pos()    const { return m_pos; }
    PONCA_MULTIARCH inline InternalType normal() const { return m_nor; }

private:
    InternalType m_pos;
    InternalType m_nor;
};



template<typename KdTreeType>
void buildKdTree(const Eigen::MatrixXd& cloudV, const Eigen::MatrixXd& cloudN, KdTreeType& tree){
    std::vector<int> ids(cloudV.rows());
    std::iota(ids.begin(), ids.end(), 0);

    using VN = std::pair<const Eigen::MatrixXd&, const Eigen::MatrixXd&>;

    // Build KdTree: do not copy coordinate but rather store Eigen::Block
    tree.buildWithSampling(VN(cloudV, cloudN),
                           ids,
                           [](VN bufs, typename KdTreeType::PointContainer &out) {
                               int s = bufs.first.rows();
                               out.reserve(s);
                               for (int i = 0; i != s; ++i)
                                   out.push_back(typename KdTreeType::DataPoint(bufs.first.row(i).transpose(),
                                                                                bufs.second.row(i).transpose()));
                           });
}