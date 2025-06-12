/*
This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <igl/opengl/glfw/Viewer.h>
#include <iostream>
#include <chrono>
#include <igl/readOBJ.h>
#include <igl/per_vertex_normals.h>

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

Eigen::MatrixXd cloudV, cloudN;
Eigen::MatrixXi meshF;

int main(int argc, char *argv[])
{
    measureTime( "[libIGL] Load Demo Mesh", []()
    {
        std::string filename = "../assets/bunny.obj"; // Works if build was made inside the working directory, this will need to be included inside the build

        igl::readOBJ(filename, cloudV, meshF);
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
    viewer.data().set_face_based(true);
    viewer.launch();
}

