/*
This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <igl/opengl/glfw/Viewer.h>
#include <igl/avg_edge_length.h>
#include <iostream>
#include <chrono>
#include <igl/readPLY.h>


#include <igl/opengl/glfw/imgui/ImGuiPlugin.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>

#include <Ponca/Fitting>
#include <Ponca/SpatialPartitioning>
#include "poncaAdapters.hpp"

#include "igl/unproject_onto_mesh.h"

using Scalar             = double;
using VectorType         = Eigen::Vector<Scalar, 3>;
using PPAdapter          = BlockPointAdapter<Scalar>;
using KdTree             = Ponca::KdTreeSparse<PPAdapter>;
using KnnGraph           = Ponca::KnnGraph<PPAdapter>;
using SmoothWeightFunc   = Ponca::DistWeightFunc<PPAdapter, Ponca::SmoothWeightKernel<Scalar> >;

enum FittingType { NONE=0, ASO, APSS, PSS };
enum DisplayedScalar { MEAN, MIN, MAX };

Eigen::MatrixXd cloudV, cloudN, cloudC; // Points position, normals and colors
Eigen::MatrixXi meshF; // The face of the mesh
igl::opengl::glfw::Viewer poncaViewer;

// Building the kdtree
KdTree tree;
constexpr float NSize        = 0.1;   /// Neighborhood size used to build the kdtree (euclidean)
constexpr int mlsIter        = 3;     /// Number of moving least squares iterations

// Kd Tree search
int k = 10;                 /// Number of neighbors to search for
int selected_pid = 0;       /// The currently selected point from the point cloud

// Some default colors
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
void colorMapPointCloudScalars(Eigen::VectorXd scalars, const bool writeLabel=true) {
    poncaViewer.data().clear_points();

    double minVal = scalars.minCoeff();
    double maxVal = scalars.maxCoeff();
    Eigen::VectorXd normalized = (scalars.array() - minVal) / (maxVal - minVal + 1e-12); // prevent divide by zero

    for (int i = 0; i < scalars.size(); ++i) {
        // Use igl::ColorMapType::JET, or other types like HOT, COOL, etc.
        Eigen::VectorXd rgb(3);
        igl::colormap(cm, normalized[i], rgb[0], rgb[1], rgb[2]);
        cloudC.row(i) = rgb;
    }
    poncaViewer.data().add_points(cloudV, cloudC);

    if (!writeLabel) return;

    // Add the labels near the points
    poncaViewer.data().clear_labels();

    for (int i = 0; i < scalars.size(); ++i) {
        std::stringstream l1;
        l1 << scalars(i) ;
        const Eigen::Vector3d offset = cloudN.row(i).transpose() * 0.0007; // Offset from the normal
        Eigen::Vector3d position     = cloudV.row(i).transpose();  // from row (1x3) to column (3x1)
        position += offset;  // apply offset

        poncaViewer.data().add_label(position, l1.str());
    }
}

/// Generic processing function: traverse point cloud and compute mean, first and second curvatures + their direction
/// \tparam FitT Defines the type of estimator used for computation
template<typename FitT>
void estimateDifferentialQuantities( DisplayedScalar displayedScalar, const bool showMinCurvatureDir = true, const bool showMaxCurvatureDir = true) {
    int nvert = tree.samples().size();

    Eigen::MatrixXd dmin( nvert, 3 ), dmax( nvert, 3 );
    Eigen::VectorXd mean ( nvert ), kmin ( nvert ), kmax ( nvert );
    Eigen::MatrixXd normal( nvert, 3 ), proj( nvert, 3 );

    measureTime( "Compute differential quantities",
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

    // Show the first and second curvature direction
    const double avg = igl::avg_edge_length(cloudV, meshF);
    if (showMinCurvatureDir) {
        poncaViewer.data().add_edges(cloudV, cloudV + dmin*avg, blue);
    }
    if (showMaxCurvatureDir) {
        poncaViewer.data().add_edges(cloudV, cloudV + dmax*avg, red);
    }

    // Display the scalar computed by the curvature estimator
    switch (displayedScalar) {
        case MEAN:
            colorMapPointCloudScalars<igl::ColorMapType::COLOR_MAP_TYPE_TURBO>(mean );
            break;
        case MIN:
            colorMapPointCloudScalars<igl::ColorMapType::COLOR_MAP_TYPE_TURBO>(kmin );
            break;
        case MAX:
            colorMapPointCloudScalars<igl::ColorMapType::COLOR_MAP_TYPE_TURBO>(kmax );
            break;
    }
}

class PluginPoncaGUI final : public igl::opengl::glfw::ViewerPlugin
{
    IGL_INLINE bool post_load() override
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
        cloudC = blue.replicate(cloudV.rows(), 1); // Color
        poncaViewer.data().add_points(cloudV, cloudC);

        // Overlay settings
        poncaViewer.data().point_size *= 0.3;
        poncaViewer.data().show_lines = false;

        return false;
    }
    IGL_INLINE bool mouse_down(int button, int /*modifier*/) override
    {
        int fid;
        cloudC = blue.replicate(cloudV.rows(), 1);
        Eigen::Vector3f bc;
        // Cast a ray in the view direction starting from the mouse position
        const double x = poncaViewer.current_mouse_x;
        const double y = poncaViewer.core().viewport(3) - poncaViewer.current_mouse_y;

        if(! igl::unproject_onto_mesh(Eigen::Vector2f(x,y), poncaViewer.core().view,
            poncaViewer.core().proj, poncaViewer.core().viewport, cloudV, meshF, fid, bc))
            return false;

        const auto& tri = meshF.row(fid);
        const auto& v0 = cloudV.row(tri[0]);
        const auto& v1 = cloudV.row(tri[1]);
        const auto& v2 = cloudV.row(tri[2]);
        const Eigen::RowVector3d query_pt = bc[0] * v0 + bc[1] * v1 + bc[2] * v2;

        selected_pid = *tree.nearest_neighbor(query_pt).begin();

        // Paint the selected point red
        cloudC.row(selected_pid) = red;
        poncaViewer.data().clear_points();
        poncaViewer.data().add_points(cloudV, cloudC);
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
    //////////////////////////////////////////////////////////

    // Load the default mesh
    std::string demo_filename = "../assets/bunny.obj";
    if (! poncaViewer.load_mesh_from_file(demo_filename))
        poncaViewer.open_dialog_load_mesh(); // If the default file was not found, we prompt the user to select a file to open

    cloudV = poncaViewer.data().V;
    meshF  = poncaViewer.data().F;
    cloudN  = poncaViewer.data().V_normals;
    cloudC = blue.replicate(cloudV.rows(), 1);

    // Check if normals have been properly loaded
    int nbUnitNormal = cloudN.rowwise().squaredNorm().sum();
    if ( nbUnitNormal != cloudV.rows() ) {
        std::cerr << "[libIGL] An error occurred when computing the normal vectors from the mesh. Aborting..."
                  << std::endl;
        return EXIT_FAILURE;
    }

    poncaViewer.core().background_color[0] = 0.12;
    poncaViewer.core().background_color[1] = 0.12;
    poncaViewer.core().background_color[2] = 0.12;
    poncaViewer.data().show_faces   = false; // Hide de face by default
    static bool showMinCurvatureDir = false; // Hide the curvatures direction by default
    static bool showMaxCurvatureDir = false;

    // Curvature estimation parameters
    static FittingType fitType = FittingType::NONE;
    static DisplayedScalar displayedScalar = DisplayedScalar::MEAN;

    // Draw additional windows
    menu.callback_draw_custom_window = [&]() {
        // Define the ponca window position and size
        ImGui::SetNextWindowPos(ImVec2(180.f * menu.menu_scaling(), 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(200, 350), ImGuiCond_FirstUseEver);
        ImGui::Begin("Ponca", nullptr, ImGuiWindowFlags_NoSavedSettings);

        // Add new group
        if (ImGui::CollapsingHeader("K-Neighbors search", ImGuiTreeNodeFlags_DefaultOpen)) {
            // The k neighbors
            if (ImGui::InputInt("k", &k))
                k = std::max(1, k);

            // ID of the currently selected point
            if (ImGui::InputInt("Selected point ID", &selected_pid))
                selected_pid = std::max(-1, selected_pid);

            // Start the neighbor search with the kdtree
            if (ImGui::Button("Colorize Neighbors", ImVec2(-1,0)) && selected_pid > 0) {
                poncaViewer.data().clear_points();
                cloudC = blue.replicate(cloudV.rows(), 1);
                cloudC.row(selected_pid) = red;

                // Paint the neighbors orange
                for(const int neighbor_idx : tree.k_nearest_neighbors(selected_pid, k)) {
                    cloudC.row(neighbor_idx) = orange;
                }

                poncaViewer.data().add_points(cloudV, cloudC);
            }
        }
        // Add new group
        if (ImGui::CollapsingHeader("Curvature estimation", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Combo("Fit type", reinterpret_cast<int*>(&fitType), "NONE\0ASO\0APSS\0PSS\0\0");
            ImGui::Combo("Scalar to display", reinterpret_cast<int*>(&displayedScalar), "MEAN\0MIN\0MAX\0\0");
            ImGui::Checkbox("Show min curvature direction", &showMinCurvatureDir);
            ImGui::Checkbox("Show max curvature direction", &showMaxCurvatureDir);

            // Update the estimation preview
            if (ImGui::Button("Update curvatures estimation")) {
                poncaViewer.data().clear_edges();

                switch (fitType) {
                    case NONE:
                        poncaViewer.data().clear_points();
                        cloudC = blue.replicate(cloudV.rows(), 1);
                        poncaViewer.data().add_points(cloudV, cloudC);
                        break;
                    case ASO:
                        std::cout << "[Ponca] ASO : ";
                        estimateDifferentialQuantities<FitASODiff>(displayedScalar, showMinCurvatureDir, showMaxCurvatureDir);
                        break;
                    case APSS:
                        std::cout << "[Ponca] APSS : ";
                        estimateDifferentialQuantities<FitAPSSDiff>(displayedScalar, showMinCurvatureDir, showMaxCurvatureDir);
                        break;
                    case PSS:
                        std::cout << "[Ponca] PSS : ";
                        estimateDifferentialQuantities<FitPlaneDiff>(displayedScalar, showMinCurvatureDir, showMaxCurvatureDir);
                        break;
                }
            }
        }
        ImGui::End();
    };

    poncaViewer.launch();
}

