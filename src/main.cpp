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

#include "igl/unproject_onto_mesh.h"

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
const Eigen::RowVector3d red(0.8,0.2,0.2), orange(0.8,0.5,0.2), blue(0.2,0.2,0.8), green(0.2,0.8,0.2);

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

template<igl::ColorMapType cm>
void colorMapPointCloudScalars(Eigen::VectorXd scalars) {
    poncaViewer.data().clear_points();

    double minVal = scalars.minCoeff();
    double maxVal = scalars.maxCoeff();
    Eigen::VectorXd normalized = (scalars.array() - minVal) / (maxVal - minVal + 1e-12); // prevent divide by zero
    Eigen::MatrixXd cloudC(scalars.size(), 3);

    for (int i = 0; i < scalars.size(); ++i) {
        // Use igl::ColorMapType::JET, or other types like HOT, COOL, etc.
        Eigen::VectorXd rgb(3);
        igl::colormap(cm, normalized[i], rgb[0], rgb[1], rgb[2]);
        cloudC.row(i) = rgb;
    }
    poncaViewer.data().add_points(cloudV, cloudC);

}
/// Generic processing function: traverse point cloud and compute mean, first and second curvatures + their direction
/// \tparam FitT Defines the type of estimator used for computation
template<typename FitT>
void estimateDifferentialQuantities( const std::string& name ) {
    int nvert = tree.samples().size();
    // const Eigen::MatrixXd cloudC = blue.replicate(cloudV.rows(), 1); // Color
    Eigen::MatrixXd dmin( nvert, 3 ), dmax( nvert, 3 );
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
    const double avg = igl::avg_edge_length(cloudV, meshF);

    poncaViewer.data().add_edges(cloudV + dmin*avg, cloudV - dmin*avg, red);
    poncaViewer.data().add_edges(cloudV + dmax*avg, cloudV - dmax*avg, blue);

    colorMapPointCloudScalars<igl::ColorMapType::COLOR_MAP_TYPE_TURBO>(mean );
}



class PluginPoncaGUI final : public igl::opengl::glfw::ViewerPlugin
{
    IGL_INLINE virtual bool post_load() override
    {
        // Clear the previous mesh when a new mesh is added, as well as the overlays
        if(poncaViewer.data_list.size() > 1)
            poncaViewer.erase_mesh(0);


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
        return false;
    }
    IGL_INLINE virtual bool mouse_down(int /*button*/, int /*modifier*/) override
    {
        return false;
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


    static int k = 10;
    int selected_pid = 0;

    // Select a curvature estimation
    enum FittingType { NONE=0, ASO, APSS, PSS};
    static FittingType fitType = NONE;
    poncaViewer.callback_mouse_down =
        [&selected_pid](igl::opengl::glfw::Viewer& viewer, int, int)->bool
        {
            int fid;
            Eigen::MatrixXd cloudC = blue.replicate(cloudV.rows(), 1);
            Eigen::Vector3f bc;
            // Cast a ray in the view direction starting from the mouse position
            double x = poncaViewer.current_mouse_x;
            double y = poncaViewer.core().viewport(3) - poncaViewer.current_mouse_y;
            if(igl::unproject_onto_mesh(Eigen::Vector2f(x,y), poncaViewer.core().view,

              poncaViewer.core().proj, poncaViewer.core().viewport, cloudV, meshF, fid, bc))
            {
                const auto& tri = meshF.row(fid);
                const auto& v0 = cloudV.row(tri[0]);
                const auto& v1 = cloudV.row(tri[1]);
                const auto& v2 = cloudV.row(tri[2]);
                Eigen::RowVector3d query_pt = bc[0] * v0 + bc[1] * v1 + bc[2] * v2;

                selected_pid = *tree.nearest_neighbor(query_pt).begin();
                std::cout << query_pt;
                // Paint the selected point red
                cloudC.row(selected_pid) = red;
                poncaViewer.data().clear_points();
                poncaViewer.data().add_points(cloudV, cloudC);


                return true;
            }
            return false;
        };
        std::cout<<R"(Usage:
      [click]  Pick face on shape

    )";
    // Add content to the default menu window
    menu.callback_draw_viewer_menu = [&]()
    {
        // Draw parent menu content
        menu.draw_viewer_menu();

        // Add new group
        if (ImGui::CollapsingHeader("Ponca", ImGuiTreeNodeFlags_DefaultOpen))
        {
            Eigen::MatrixXd cloudC = blue.replicate(cloudV.rows(), 1);

            // We can also use a std::vector<std::string> defined dynamically
            if (ImGui::InputInt("k", &k)) {
                k = std::max(1, k);
            }
            // We can also use a std::vector<std::string> defined dynamically
            if (ImGui::InputInt("Selected point id", &selected_pid)) {
                selected_pid = std::max(-1, selected_pid);
            }


            if (ImGui::Button("Colorize Neighbors", ImVec2(-1,0)) && selected_pid > 0)
            {
                poncaViewer.data().clear_points();

                cloudC.row(selected_pid) = red;

                // Paint the neighbors orange
                for(int neighbor_idx : tree.k_nearest_neighbors(selected_pid, k)) {
                    cloudC.row(neighbor_idx) = orange;
                }

                poncaViewer.data().add_points(cloudV, cloudC);
            }

            ImGui::Combo("Fit type", reinterpret_cast<int*>(&fitType), "NONE\0ASO\0APSS\0PSS\0\0");

            // Update the estimation preview
            if (ImGui::Button("Update curvatures")) {
                poncaViewer.data().clear_edges();
                switch (fitType) {
                    case NONE:
                        poncaViewer.data().clear_points();
                        poncaViewer.data().add_points(cloudV, cloudC);
                        break;
                    case ASO:
                        estimateDifferentialQuantities<FitASODiff>("ASO");
                        break;
                    case APSS:
                        estimateDifferentialQuantities<FitAPSSDiff>("APSS");
                        break;
                    case PSS:
                        estimateDifferentialQuantities<FitPlaneDiff>("PSS");
                        break;
                }
            }
        }
    };

    poncaViewer.launch();
}

