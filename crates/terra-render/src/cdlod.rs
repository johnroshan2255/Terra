//! CDLOD -- Continuous Distance-Dependent Level of Detail.
//!
//! A quadtree over the world selects a LOD per node; every node draws the same
//! instanced grid mesh and morphs its vertices toward the parent LOD by camera
//! distance, so adjacent levels meet without cracks and without stitch meshes.
//!
//! Heights are sampled from a texture in the vertex shader. Per-chunk vertex
//! buffers are never uploaded.

// TODO: quadtree select -> instance buffer -> one draw per LOD ring.
