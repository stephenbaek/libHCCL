Wulff Flow Module
=================

A class for simulating the Wulff flow on a discrete surface.

The code is based on S. Baek et al., "The fast Wulff flow on discrete manifolds."

## Usage ##
``` cpp
// Initialization
hccl::Wulff wulff;
wulff.set_values(levelset);
wulff.set_geometry(mesh);
wulff.set_timestep(200);  // optional
wulff.set_beta(beta);     // optional
wulff.initialize();
    
// Runtime
wulff.solve();        // simulate one step
```

## Caution ##
- set_***() must be called before initialize()
