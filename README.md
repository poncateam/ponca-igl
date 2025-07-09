# ponca-igl
This demo application was built using :

* **[Ponca](https://github.com/poncateam/ponca) :** for POiNt Cloud Analysis and acceleration structures (curvature estimation and kdtree)

*  **[libIGL](https://github.com/libigl/libigl) :** for the GUI and data loading

## Keyboard and mouse controls

* **Left and right mouse click :** move around the scene

* **Middle mouse click :** selects a point (highlighted in red)

## Menu

### Usefull menu option in the "Viewer" sub-menu 

> Mesh
> * **Load :** loads a point cloud (obj, vtk, ply)

> Viewing options
> * **Orthographic view :** Disables the perspective

> Overlays
> * **Show extra label :** Show the scalar value, for each points, that was computed by the curvature estimator (either mean, kmin, kmax)

### Usefull menu option in the "Ponca" sub-menu

> K-Neighbors search
> * **k :** The number of neighbors that are going to be highlighted in orange
> * **Selected point ID :** The index of the currently selected point (The point can be selected directly on the point cloud with the middle mouse click)

> Curvature estimation
> * **Fit type :** The Fitting that is used to compute the curvatures (see [Ponca Fit Doc](https://poncateam.github.io/ponca/fitting.html)). Can be either :
>   - [PSS](https://dl.acm.org/doi/10.5555/601671.601673) : Point Set Surface
>   - [APSS](https://dl.acm.org/doi/10.1145/1276377.1276406) : Algebraic Point Set Surfaces
>   - [ASO](https://onlinelibrary.wiley.com/doi/10.1111/cgf.14368) : Algebraic Shape Operator