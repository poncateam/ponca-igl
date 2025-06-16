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
igl::opengl::glfw::Viewer poncaViewer;

float NSize        = 0.1;   /// < neighborhood size (euclidean)
Scalar pointRadius = 0.005; /// < display radius of the point cloud
int mlsIter        = 3;     /// < number of moving least squares iterations
const Eigen::RowVector3d red(0.8,0.2,0.2), orange(0.8,0.5,0.2), blue(0.2,0.2,0.8);

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

class PluginPoncaGUI : public igl::opengl::glfw::ViewerPlugin
{
    IGL_INLINE virtual bool post_load() override
    {
        // TODO : clear the previous mesh when a new mesh is added, as well as the overlays
        poncaViewer.data().clear_points();
        poncaViewer.data().clear_edges();

        // Retrieve mesh information
        cloudV = poncaViewer.data().V;
        meshF  = poncaViewer.data().F;
        cloudN  = poncaViewer.data().V_normals;

        // Build Ponca KdTree
        measureTime( "[Ponca] Build KdTree", []() {
            buildKdTree(cloudV, cloudN, tree);
        });

        // Display point clouds
        const Eigen::MatrixXd cloudC = blue.replicate(cloudV.rows(), 1); // Color
        poncaViewer.data().add_points(cloudV, cloudC);

        // Overlay settings
        poncaViewer.data().point_size *= 0.3;
        poncaViewer.data().show_lines = false;
        return true;
    }
};

int main(int argc, char *argv[])
{
    // Attach a plugin
    igl::opengl::glfw::imgui::ImGuiPlugin plugin;
    poncaViewer.plugins.push_back(&plugin);
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    plugin.widgets.push_back(&menu);

    PluginPoncaGUI pluginPonca;
    poncaViewer.plugins.push_back(&pluginPonca);

    ////////// Fitting type for curvature estimation //////////
    using FitPlane = Ponca::Basket<PPAdapter, SmoothWeightFunc, Ponca::CovariancePlaneFit>;
    using FitPlaneDiff = Ponca::BasketDiff<
            FitPlane,
            Ponca::DiffType::FitSpaceDer,
            Ponca::CovariancePlaneDer,
            Ponca::CurvatureEstimatorBase, Ponca::NormalDerivativesCurvatureEstimator>;

    using FitAPSS = Ponca::Basket<PPAdapter, SmoothWeightFunc, Ponca::OrientedSphereFit>;
    using FitAPSSDiff = Ponca::BasketDiff<
            FitAPSS,
            Ponca::DiffType::FitSpaceDer,
            Ponca::OrientedSphereDer,
            Ponca::CurvatureEstimatorBase, Ponca::NormalDerivativesCurvatureEstimator>;

    using FitASO = FitAPSS;
    using FitASODiff = Ponca::BasketDiff<
            FitASO,
            Ponca::DiffType::FitSpaceDer,
            Ponca::OrientedSphereDer, Ponca::MlsSphereFitDer,
            Ponca::CurvatureEstimatorBase, Ponca::NormalDerivativesCurvatureEstimator>;

    ////////////////////////////////////////

    std::string demo_filename = "../assets/bunny.obj";
    // Plot the mesh

    if (! poncaViewer.load_mesh_from_file(demo_filename))
    {
        poncaViewer.open_dialog_load_mesh();
    }
    cloudV = poncaViewer.data().V;
    meshF  = poncaViewer.data().F;
    cloudN  = poncaViewer.data().V_normals;

    // Check if normals have been properly loaded
    int nbUnitNormal = cloudN.rowwise().squaredNorm().sum();
    if ( nbUnitNormal != cloudV.rows() ) {
        std::cerr << "[libIGL] An error occurred when computing the normal vectors from the mesh. Aborting..."
                  << std::endl;
        return EXIT_FAILURE;
    }


    int nvert = tree.samples().size();
    Eigen::MatrixXd dmin( nvert, 3 ), dmax( nvert, 3 );
    const double avg = igl::avg_edge_length(cloudV, meshF);
    static int k = 10;

    // Select a curvature estimation
    enum FittingType { ASO=0, APSS, PSS};
    static FittingType fitType = PSS;

    // Add content to the default menu window
    menu.callback_draw_viewer_menu = [&]()
    {
        // Draw parent menu content
        menu.draw_viewer_menu();

        // Add new group
        if (ImGui::CollapsingHeader("Ponca", ImGuiTreeNodeFlags_DefaultOpen))
        {

            // We can also use a std::vector<std::string> defined dynamically
            static std::vector<std::string> choices;
            static int idx_choice = 0;
            if (ImGui::InputInt("k", &k))
            {
                k = std::max(1, k);
            }
            ImGui::Combo("k", &idx_choice, choices);


            if (ImGui::Button("Colorise Neighbors", ImVec2(-1,0)))
            {
                poncaViewer.data().clear_points();

                Eigen::MatrixXd cloudC = blue.replicate(cloudV.rows(), 1);
                cloudC.row(0) = red;

                for(int neighbor_idx : tree.k_nearest_neighbors(0, k)) {
                    cloudC.row(neighbor_idx) = orange;
                }

                poncaViewer.data().add_points(cloudV, cloudC);
            }

            ImGui::Combo("Type", (int *)(&fitType), "ASO\0APSS\0PSS\0\0");

            // Update the estimation preview
            if (ImGui::Button("Update curvatures"))
            {
                poncaViewer.data().clear_edges();
                switch (fitType)
                {
                case ASO:
                    estimateDifferentialQuantities<FitASODiff>("ASO", dmin, dmax);
                    break;
                case APSS:
                    estimateDifferentialQuantities<FitAPSSDiff>("APSS", dmin, dmax);
                    break;
                case PSS:
                    estimateDifferentialQuantities<FitPlaneDiff>("PSS", dmin, dmax);
                    break;
                }
                poncaViewer.data().add_edges(cloudV + dmin*avg, cloudV - dmin*avg, red);
                poncaViewer.data().add_edges(cloudV + dmax*avg, cloudV - dmax*avg, blue);
            }
        }
    };



    poncaViewer.launch();
}

