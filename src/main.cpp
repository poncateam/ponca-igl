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

#include "igl/unproject_ray.h"

using Scalar             = double;
using VectorType         = Eigen::Vector<Scalar, 3>;
using PPAdapter          = BlockPointAdapter<Scalar>;
using KdTree             = Ponca::KdTreeSparse<PPAdapter>;
using KnnGraph           = Ponca::KnnGraph<PPAdapter>;
using SmoothWeightFunc   = Ponca::DistWeightFunc<PPAdapter, Ponca::SmoothWeightKernel<Scalar> >;

enum FittingType { ASO, APSS, PSS, UnorientedSphere };
enum DisplayedScalar { NONE=0, MEAN, MIN, MAX };

Eigen::MatrixXd cloudV, cloudN, cloudC, cloudP; // Points position, normals, colors and project values
Eigen::MatrixXi meshF; // The face of the mesh
igl::opengl::glfw::Viewer poncaViewer;

bool displayProjPos = false;
Eigen::MatrixXd getPointCloudPosition() {
    return displayProjPos ? cloudP : cloudV;
}

// Building the kdtree
KdTree tree;
float NSize           = 0.1;   /// Neighborhood size used in the curvature estimation (and to build the kdtree)
int mlsIter           = 3;     /// Number of moving least squares iterations
float pointSize       = 10;

// Kd Tree search
int k = 10;                 /// Number of neighbors to search for
int selected_pid = 0;       /// The currently selected point from the point cloud

// Some default colors
const Eigen::RowVector3d red(0.8,0.2,0.2), orange(0.8,0.5,0.2), blue(0.2,0.2,0.8), green(0.2,0.8,0.2), cyan(0.2,0.8,0.8);

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
void processPointCloud(Functor f){
#pragma omp parallel for
    for (int i = 0; i < tree.samples().size(); ++i) {
        VectorType pos = tree.points()[i].pos();

        for( int mm = 0; mm < mlsIter; ++mm) {
            FitT fit;
            fit.setWeightFunc({pos, NSize});
            fit.init();

            processRangeNeighbors(i, [&fit](int j){
                fit.addNeighbor(tree.points()[j]);
            });

            if (fit.finalize() == Ponca::STABLE){
                pos = fit.project( pos );
                if ( mm == mlsIter -1 ) {
                    // last mls step, calling functor
                    f(i, fit, pos);
                    cloudP.row(i) = pos.transpose();
                }
            }
            else {
                std::cerr << "Warning: fit " << i << " is not stable" << std::endl;
                break;
            }
        }
    }
}

/// Recolorize the point cloud on the viewer given a set of scalar values (associated to each point) and a colormap. Can also write the values next to the point cloud (extra label)
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
    poncaViewer.data().add_points(getPointCloudPosition(), cloudC);

    if (!writeLabel) return;

    // Add the labels near the points
    poncaViewer.data().clear_labels();

    for (int i = 0; i < scalars.size(); ++i) {
        std::stringstream l1;
        l1 << scalars(i) ;
        const Eigen::Vector3d offset = cloudN.row(i).transpose() * 0.0007; // Offset from the normal
        Eigen::Vector3d position     = getPointCloudPosition().row(i).transpose();  // from row (1x3) to column (3x1)
        position += offset;  // apply offset

        poncaViewer.data().add_label(position, l1.str());
    }
}

/// Generic processing function: traverse point cloud and compute mean, first and second curvatures + their direction
/// \tparam FitT Defines the type of estimator used for computation
template<typename FitT>
void estimateDifferentialQuantities( DisplayedScalar displayedScalar, const bool showMinCurvatureDir = true, const bool showMaxCurvatureDir = true, const bool showNormal = false) {
    int nvert = tree.samples().size();

    Eigen::MatrixXd dmin( nvert, 3 ), dmax( nvert, 3 );
    Eigen::VectorXd mean ( nvert ), kmin ( nvert ), kmax ( nvert );
    Eigen::MatrixXd normal( nvert, 3 ), proj( nvert, 3 );

    measureTime( "Compute differential quantities",
                 [&mean, &kmin, &kmax, &normal, &dmin, &dmax, &proj]() {
        processPointCloud<FitT>([&mean, &kmin, &kmax, &normal, &dmin, &dmax, &proj]
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
        poncaViewer.data().add_edges(getPointCloudPosition(), getPointCloudPosition() + dmin*avg, blue);
    }
    if (showMaxCurvatureDir) {
        poncaViewer.data().add_edges(getPointCloudPosition(), getPointCloudPosition() + dmax*avg, red);
    }
    if (showNormal) {
        poncaViewer.data().add_edges(getPointCloudPosition(), getPointCloudPosition()+normal*avg, cyan);
    }

    // Display the scalar computed by the curvature estimator
    switch (displayedScalar) {
        case NONE:
            poncaViewer.data().clear_points();
            cloudC = blue.replicate(cloudV.rows(), 1);
            poncaViewer.data().add_points(getPointCloudPosition(), cloudC);
            break;
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
        cloudV  = poncaViewer.data().V;
        cloudP  = poncaViewer.data().V;
        meshF   = poncaViewer.data().F;
        cloudN  = poncaViewer.data().V_normals;

        // Build Ponca KdTree
        measureTime( "[Ponca] Build KdTree", []() {
            buildKdTree(cloudV, cloudN, tree);
        });

        // Display point clouds
        cloudC = blue.replicate(cloudV.rows(), 1); // Color
        poncaViewer.data().add_points(cloudV, cloudC);

        // Overlay settings
        poncaViewer.data().point_size = pointSize;
        poncaViewer.data().show_lines = false;

        return false;
    }
    IGL_INLINE bool mouse_down(int button, int /*modifier*/) override
    {
        // Select only if middle click
        if (button != 1)
            return false;

        int fid;
        cloudC = blue.replicate(cloudV.rows(), 1);
        Eigen::Vector3f bc;
        // Cast a ray in the view direction starting from the mouse position
        const double x = poncaViewer.current_mouse_x;
        const double y = poncaViewer.core().viewport(3) - poncaViewer.current_mouse_y;

        // Creates the ray in the world space
        Eigen::Vector3f origin,dir;
        igl::unproject_ray(
            Eigen::Vector2f(x,y), poncaViewer.core().view,
            poncaViewer.core().proj, poncaViewer.core().viewport,
            origin,dir
        );
        dir.normalize();

        float minDist = std::numeric_limits<float>::max();

        // Searching for the closest point to the ray
        for (int i = 0; i < cloudV.rows(); ++i) {
            Eigen::Vector3f vertex = getPointCloudPosition().row(i).cast<float>();
            Eigen::Vector3f v = vertex - origin;
            float dotProd = v.dot(dir);

            if (dotProd < 0) continue; // The vertex is behind the ray camera

            Eigen::Vector3f closestPointOnRay = origin + dotProd * dir;
            float distanceToRay = (vertex - closestPointOnRay).norm();

            if (distanceToRay < minDist) {
                selected_pid = i;
                minDist = distanceToRay;
            }
        }

        // No vertex found in front of the camera
        if (minDist == std::numeric_limits<float>::max())
            return false;

        // Paint the selected point red
        cloudC.row(selected_pid) = red;
        poncaViewer.data().clear_points();
        poncaViewer.data().add_points(getPointCloudPosition(), cloudC);
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

    using FitUnorientedSphere = Ponca::BasketDiff<
                Ponca::Basket<PPAdapter, SmoothWeightFunc, Ponca::UnorientedSphereFit>,
                Ponca::DiffType::FitSpaceDer,
                Ponca::UnorientedSphereDer,
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
    static bool showFitGradientDir  = false;

    // Curvature estimation parameters
    static FittingType fitType = FittingType::ASO;
    static DisplayedScalar displayedScalar = DisplayedScalar::MEAN;
    menu.callback_draw_viewer_menu = [&]()
    {
        // Draw parent menu content
        menu.draw_viewer_menu();
        if (ImGui::DragFloat("Point size", &pointSize))
            poncaViewer.data().point_size = pointSize;
    };

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

                poncaViewer.data().add_points(getPointCloudPosition(), cloudC);
            }
        }
        // Add new group
        if (ImGui::CollapsingHeader("Curvature estimation", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::DragFloat("Fitting radius", &NSize, 0.001f, 0.001f))
                NSize = std::max(NSize, 0.001f);
            ImGui::InputInt("Number of MLS iteration", &mlsIter);
            ImGui::Combo("Fit type", reinterpret_cast<int*>(&fitType), "ASO\0APSS\0PSS\0UnorientedSphere\0\0");
            ImGui::Combo("Scalar to display", reinterpret_cast<int*>(&displayedScalar), "NONE\0MEAN\0MIN\0MAX\0\0");
            ImGui::Checkbox("Show min curvature direction"  , &showMinCurvatureDir);
            ImGui::Checkbox("Show max curvature direction"  , &showMaxCurvatureDir);
            ImGui::Checkbox("Show fit gradient direction"   , &showFitGradientDir);
            ImGui::Checkbox("Display the projected position", &displayProjPos);

            // Update the estimation preview
            if (ImGui::Button("Update curvatures estimation")) {
                poncaViewer.data().clear_edges();

                switch (fitType) {
                    case ASO:
                        std::cout << "[Ponca] ASO : ";
                        estimateDifferentialQuantities<FitASODiff>(displayedScalar, showMinCurvatureDir, showMaxCurvatureDir, showFitGradientDir);
                        break;
                    case APSS:
                        std::cout << "[Ponca] APSS : ";
                        estimateDifferentialQuantities<FitAPSSDiff>(displayedScalar, showMinCurvatureDir, showMaxCurvatureDir, showFitGradientDir);
                        break;
                    case PSS:
                        std::cout << "[Ponca] PSS : ";
                        estimateDifferentialQuantities<FitPlaneDiff>(displayedScalar, showMinCurvatureDir, showMaxCurvatureDir, showFitGradientDir);
                        break;
                    case UnorientedSphere:
                        std::cout << "[Ponca] UnorientedSphere : ";
                        estimateDifferentialQuantities<FitUnorientedSphere>(displayedScalar, showMinCurvatureDir, showMaxCurvatureDir);
                        break;
                }
            }
        }
        ImGui::End();
    };

    poncaViewer.launch();
}

