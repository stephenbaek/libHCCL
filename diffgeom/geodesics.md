Geodesic Distance Module
========================

A class for computing the geodesic distance on a discrete surface.

The code is based on K. Crane et al., "Geodesics in Heat," ACM TOG, 2013.

## Usage ##
``` cpp
// Initialization
hccl::Geodesics geod;
geod.set_geometry(mesh);
geod.set_timestep(h*average_edge_length*average_edge_length);
geod.solve(source_vertex_id);

// Compute the geodesic distance to i-th vertex from the source vertex
double dist = geod.get_distance(i);
```

## Caution ##
- set_geometry() must be called before everything!
- A geometry data and a time step must be specified before to call solve();
