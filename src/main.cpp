/*
This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <igl/opengl/glfw/Viewer.h>
#include <igl/avg_edge_length.h>
#include <iostream>
#include <chrono>
#include <igl/readOBJ.h>
#include <igl/readPLY.h>
#include <igl/per_vertex_normals.h>


#include <igl/opengl/glfw/imgui/ImGuiPlugin.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>

#include <Ponca/Fitting>
#include <Ponca/SpatialPartitioning>
#include "poncaAdapters.hpp"

#include <igl/principal_curvature.h>

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

template <typename Functor>
void processRangeNeighbors(int i, Functor f){
    for (int j : tree.range_neighbors(i, NSize)){
        f(j);
    }
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
void estimateDifferentialQuantities(const std::string& name, Eigen::MatrixXd& dmin, Eigen::MatrixXd& dmax ) {
    int nvert = tree.samples().size();
    Eigen::VectorXd mean ( nvert ), kmin ( nvert ), kmax ( nvert );
    Eigen::MatrixXd normal( nvert, 3 ), proj( nvert, 3 );

    measureTime( "[Ponca] Compute differential quantities using " + name,
                 [&mean, &kmin, &kmax, &normal, &dmin, &dmax, &proj]() {
        processPointCloud<FitT>(SmoothWeightFunc(NSize),
                                [&mean, &kmin, &kmax, &normal, &dmin, &dmax, &proj]
                                ( const int i, const FitT& fit, const VectorType& mlsPos){

            mean(i) = fit.kMean();
            kmax(i) = fit.kmax();
            kmin(i) = fit.kmin();

            normal.row( i ) = fit.primitiveGradient();
            dmin.row( i )   = fit.kminDirection();
            dmax.row( i )   = fit.kmaxDirection();

            proj.row( i )   = mlsPos - tree.points()[i].pos();
        });
    });
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

    // Attach a menu plugin
    igl::opengl::glfw::imgui::ImGuiPlugin plugin;
    viewer.plugins.push_back(&plugin);
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    plugin.widgets.push_back(&menu);

    const Eigen::RowVector3d red(0.8,0.2,0.2), orange(0.8,0.5,0.2), blue(0.2,0.2,0.8);
    static int k = 10;

    // Add content to the default menu window
    menu.callback_draw_viewer_menu = [&]()
    {
        // Draw parent menu content
        menu.draw_viewer_menu();

        // Add new group
        if (ImGui::CollapsingHeader("Ponca", ImGuiTreeNodeFlags_DefaultOpen))
        {
          // Expose an enumeration type
          enum Orientation { Up=0, Down, Left, Right };
          static Orientation dir = Up;
          ImGui::Combo("Type", (int *)(&dir), "Up\0Down\0Left\0Right\0\0");

          // We can also use a std::vector<std::string> defined dynamically
          static std::vector<std::string> choices;
          static int idx_choice = 0;
          if (ImGui::InputInt("k", &k))
          {
            k = std::max(1, k);
          }
           ImGui::Combo("Letter", &idx_choice, choices);
        }
    };



    viewer.data().set_mesh(cloudV, meshF);
    viewer.data().show_lines = false;

    // Build Ponca KdTree
    measureTime( "[Ponca] Build KdTree", []() {
        buildKdTree(cloudV, cloudN, tree);
    });

    Eigen::MatrixXd cloudC = blue.replicate(cloudV.rows(), 1);
    cloudC.row(0) = red;

    for(int neighbor_idx : tree.k_nearest_neighbors(0, k)) {
        cloudC.row(neighbor_idx) = orange;
    }

    int nvert = tree.samples().size();
    Eigen::MatrixXd dmin( nvert, 3 ), dmax( nvert, 3 );
    const double avg = igl::avg_edge_length(cloudV, meshF);

    viewer.data().add_points(cloudV, cloudC);

    // Curvature estimation
    using FitPlane = Ponca::Basket<PPAdapter, SmoothWeightFunc, Ponca::CovariancePlaneFit>;
    using FitPlaneDiff = Ponca::BasketDiff<
        FitPlane,
        Ponca::DiffType::FitSpaceDer,
        Ponca::CovariancePlaneDer,
        Ponca::CurvatureEstimatorBase, Ponca::NormalDerivativesCurvatureEstimator>;
    estimateDifferentialQuantities<FitPlaneDiff>("PSS", dmin, dmax);
    viewer.data().add_edges(cloudV + dmin*avg, cloudV - dmin*avg, red);
    viewer.data().add_edges(cloudV + dmax*avg, cloudV - dmax*avg, blue);

    viewer.launch();
}

