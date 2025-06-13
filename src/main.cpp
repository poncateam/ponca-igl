/*
This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <igl/opengl/glfw/Viewer.h>
#include <iostream>
#include <chrono>
#include <igl/readOBJ.h>
#include <igl/readPLY.h>
#include <igl/per_vertex_normals.h>

#include <Ponca/Fitting>
#include <Ponca/SpatialPartitioning>
#include "poncaAdapters.hpp"

using Scalar             = double;
using VectorType         = Eigen::Vector<Scalar, 3>;
using PPAdapter          = BlockPointAdapter<Scalar>;
using KdTree             = Ponca::KdTreeSparse<PPAdapter>;
using KnnGraph           = Ponca::KnnGraph<PPAdapter>;
using SmoothWeightFunc   = Ponca::DistWeightFunc<PPAdapter, Ponca::SmoothWeightKernel<Scalar> >;


// KdTree tree;
Eigen::MatrixXd cloudV, cloudN;
Eigen::MatrixXi meshF;
KdTree tree;
KnnGraph* knnGraph {nullptr};

float NSize        = 0.1;   /// < neighborhood size (euclidean)
Scalar pointRadius = 0.005; /// < display radius of the point cloud
int mlsIter        = 3;     /// < number of moving least squares iterations

/// Convenience function measuring and printing the processing time of F
template <typename Functor>
void measureTime( const std::string &actionName, Functor F ){
    using namespace std::literals; // enables the usage of 24h instead of e.g. std::chrono::hours(24)

    const std::chrono::time_point<std::chrono::steady_clock> start =
            std::chrono::steady_clock::now();
    F(); // run process
    const auto end = std::chrono::steady_clock::now();
    std::cout << actionName << " in " << (end - start) / 1ms << "ms.\n";
}

/// Generic processing function: traverse point cloud, compute fitting, and use functor to process fitting output
/// \note Functor is called only if fit is stable
template<typename FitT, typename Functor>
void processPointCloud(const typename FitT::WeightFunction& w, Functor f){
#pragma omp parallel for
    for (int i = 0; i < tree.samples().size(); ++i) {
        VectorType pos = tree.points()[i].pos();

        for( int mm = 0; mm < mlsIter; ++mm) {
            FitT fit;
            fit.setWeightFunc(w);
            fit.init( pos );

            processRangeNeighbors(i, [&fit](int j){
                fit.addNeighbor(tree.points()[j]);
            });

            if (fit.finalize() == Ponca::STABLE){
                pos = fit.project( pos );
                if ( mm == mlsIter -1 ) // last mls step, calling functor
                    f(i, fit, pos);
            }
            else {
                std::cerr << "Warning: fit " << i << " is not stable" << std::endl;
                break;
            }
        }
    }
}

/// Generic processing function: traverse point cloud and compute mean, first and second curvatures + their direction
/// \tparam FitT Defines the type of estimator used for computation
template<typename FitT>
void estimateDifferentialQuantities_impl(const std::string& name) {
    int nvert = tree.samples().size();
    Eigen::VectorXd mean ( nvert ), kmin ( nvert ), kmax ( nvert );
    Eigen::MatrixXd normal( nvert, 3 ), dmin( nvert, 3 ), dmax( nvert, 3 ), proj( nvert, 3 );

    measureTime( "[Ponca] Compute differential quantities using " + name,
                 [&mean, &kmin, &kmax, &normal, &dmin, &dmax, &proj]() {
        processPointCloud<FitT>(SmoothWeightFunc(NSize),
                                [&mean, &kmin, &kmax, &normal, &dmin, &dmax, &proj]
                                ( int i, const FitT& fit, const VectorType& mlsPos){

            mean(i) = fit.kMean();
            kmax(i) = fit.kmax();
            kmin(i) = fit.kmin();

            normal.row( i ) = fit.primitiveGradient();
            dmin.row( i )   = fit.kminDirection();
            dmax.row( i )   = fit.kmaxDirection();

            proj.row( i )   = mlsPos - tree.points()[i].pos();
        });
    });

    // measureTime( "[Polyscope] Update differential quantities",
    //      [&name, &mean, &kmin, &kmax, &normal, &dmin, &dmax, &proj]() {
    //          cloud->addScalarQuantity(name + " - Mean Curvature", mean)->setMapRange({-10,10});
    //          cloud->addScalarQuantity(name + " - K1", kmin)->setMapRange({-10,10});
    //          cloud->addScalarQuantity(name + " - K2", kmax)->setMapRange({-10,10});
    //
    //          cloud->addVectorQuantity(name + " - normal", normal)->setVectorLengthScale(
    //                  Scalar(2) * pointRadius);
    //          cloud->addVectorQuantity(name + " - K1 direction", dmin)->setVectorLengthScale(
    //                  Scalar(2) * pointRadius);
    //          cloud->addVectorQuantity(name + " - K2 direction", dmax)->setVectorLengthScale(
    //                  Scalar(2) * pointRadius);
    //          cloud->addVectorQuantity(name + " - projection", proj, polyscope::VectorType::AMBIENT);
    // });
}


bool endsWith (std::string const &fullString, std::string const &ending) {
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}

void readMesh( const std::string filename) {
    if (endsWith(filename, ".obj"))
    {
        igl::readOBJ(filename, cloudV, meshF);
    } else if (endsWith(filename, ".ply"))
    {
        igl::readPLY(filename, cloudV, meshF);
    }
    igl::per_vertex_normals(cloudV, meshF, cloudN);
}

int main(int argc, char *argv[])
{
    measureTime( "[libIGL] Load Demo Mesh", []()
    {
        std::string filename = "../assets/bunny.obj";
        // std::string filename = "../assets/GrosNuage30M.ply";
        readMesh(filename);
        igl::per_vertex_normals(cloudV, meshF, cloudN);
    } );


    // Check if normals have been properly loaded
    int nbUnitNormal = cloudN.rowwise().squaredNorm().sum();
    if ( nbUnitNormal != cloudV.rows() ) {
        std::cerr << "[libIGL] An error occurred when computing the normal vectors from the mesh. Aborting..."
                  << std::endl;
        return EXIT_FAILURE;
    }

    // Plot the mesh
    igl::opengl::glfw::Viewer viewer;
    viewer.data().set_mesh(cloudV, meshF);
    // viewer.data().set_face_based(true);

    // Build Ponca KdTree
    measureTime( "[Ponca] Build KdTree", []() {
        buildKdTree(cloudV, cloudN, tree);
    });

    // Hide wireframe
    viewer.data().show_lines = false;
    viewer.data().add_points(cloudV, Eigen::RowVector3d(0, 0, 0));
    viewer.launch();

    //Bounding Box (used in the slicer)
    // lower = cloudV.colwise().minCoeff();
    // upper = cloudV.colwise().maxCoeff();
}

