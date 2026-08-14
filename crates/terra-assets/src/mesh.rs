//! glTF import for rocks, trees, and props.
//!
//! Deliberately not a custom mesh format. glTF means every DCC tool can author
//! content for the editor without a bespoke exporter, and it is what the free
//! asset libraries hand you: Poly Haven serves `.gltf`, so a downloaded rock
//! drops into `assets/models/` and appears in the palette next run.
//!
//! Meshes that fail to load, and an empty folder, both fall back to the
//! generated species below. A scatter tool with nothing to scatter cannot be
//! evaluated, and "install these files first" is a bad first run.

use anyhow::{Context, Result};
use rayon::prelude::*;
use std::path::{Path, PathBuf};

/// Running sum for one grid cell during clustering.
#[derive(Default, Clone, Copy)]
struct Bucket {
    pos: [f64; 3],
    normal: [f64; 3],
    uv: [f64; 2],
    count: f64,
}

/// One clustering result, kept whole so the search can keep the best.
struct Cluster {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    indices: Vec<u32>,
}

/// A decoded albedo map: RGBA8, sRGB-encoded.
#[derive(Clone)]
pub struct Texture {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

impl std::fmt::Debug for Texture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Texture({}x{})", self.width, self.height)
    }
}

/// One drawable shape in the renderer's own terms.
#[derive(Clone, Debug, Default)]
pub struct MeshData {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub indices: Vec<u32>,
    /// Linear-space albedo, taken from the glTF base colour factor. Used as a
    /// tint, and as the whole colour when there is no map.
    pub base_color: [f32; 3],
    pub albedo: Option<Texture>,
    /// Fragments below this alpha are discarded. Scanned foliage is leaf cards
    /// with a cut-out mask; drawn opaque it is a box of triangles.
    pub alpha_cutoff: Option<f32>,
    /// Leaf cards have to be visible from behind. Back-face culling a scanned
    /// plant removes half of every leaf.
    pub double_sided: bool,
}

impl MeshData {
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Radius of a sphere at the origin containing every vertex. The scatter
    /// pass needs it to size the cell bounds it culls against.
    pub fn bounding_radius(&self) -> f32 {
        self.positions
            .iter()
            .map(|p| (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt())
            .fold(0.0, f32::max)
    }

    /// Collapse vertices onto a coarse grid until the mesh is under
    /// `target_tris`.
    ///
    /// Scanned assets are film-resolution -- a Poly Haven sapling is 433k
    /// triangles, and scattering ten thousand of those is 4 billion. Vertex
    /// clustering is the crude decimator: bucket vertices by cell, average
    /// each bucket, drop the triangles that collapse to a line. It does not
    /// preserve silhouettes the way edge-collapse does, which matters for a
    /// hero asset and does not for something the size of a thumbnail on screen.
    pub fn decimate(&mut self, target_tris: usize) {
        if self.triangle_count() <= target_tris || self.positions.is_empty() {
            return;
        }
        let (mut lo_b, mut hi_b) = ([f32::MAX; 3], [f32::MIN; 3]);
        for p in &self.positions {
            for k in 0..3 {
                lo_b[k] = lo_b[k].min(p[k]);
                hi_b[k] = hi_b[k].max(p[k]);
            }
        }
        let extent = (0..3).map(|k| hi_b[k] - lo_b[k]).fold(0.0f32, f32::max).max(1e-6);

        // Binary search for the *finest* grid that meets the target. Guessing a
        // grid from the reduction ratio and accepting the first result under
        // budget approaches from the wrong side: it overshoots, and a 100k
        // triangle trunk comes out at 300 triangles.
        let (mut lo, mut hi) = (3usize, 512usize);
        let mut best: Option<Cluster> = None;
        while lo <= hi {
            let mid = (lo + hi) / 2;
            let c = self.cluster(mid, lo_b, extent);
            if c.indices.len() / 3 <= target_tris {
                lo = mid + 1;
                best = Some(c);
            } else {
                if mid == 0 {
                    break;
                }
                hi = mid - 1;
            }
        }

        if let Some(c) = best {
            self.positions = c.positions;
            self.normals = c.normals;
            self.uvs = c.uvs;
            self.indices = c.indices;
        }
    }

    /// Collapse vertices onto a `grid`-cell lattice and remap the triangles.
    fn cluster(&self, grid: usize, lo: [f32; 3], extent: f32) -> Cluster {
        let cell = extent / grid as f32;
        let key = |p: &[f32; 3]| {
            (
                ((p[0] - lo[0]) / cell) as u32,
                ((p[1] - lo[1]) / cell) as u32,
                ((p[2] - lo[2]) / cell) as u32,
            )
        };
        let mut slot: std::collections::HashMap<(u32, u32, u32), u32> =
            std::collections::HashMap::new();
        let mut remap = vec![0u32; self.positions.len()];
        let mut acc: Vec<Bucket> = Vec::new();

        for (i, p) in self.positions.iter().enumerate() {
            let k = key(p);
            let next = acc.len() as u32;
            let idx = *slot.entry(k).or_insert(next);
            if idx as usize == acc.len() {
                acc.push(Bucket::default());
            }
            let a = &mut acc[idx as usize];
            for c in 0..3 {
                a.pos[c] += p[c] as f64;
                a.normal[c] += self.normals.get(i).map(|n| n[c]).unwrap_or(0.0) as f64;
            }
            for c in 0..2 {
                a.uv[c] += self.uvs.get(i).map(|u| u[c]).unwrap_or(0.0) as f64;
            }
            a.count += 1.0;
            remap[i] = idx;
        }

        // Triangles are independent once the remap exists, and a scanned mesh
        // has hundreds of thousands of them per search step.
        let indices: Vec<u32> = self
            .indices
            .par_chunks_exact(3)
            .filter_map(|t| {
                let (a, b, c) = (remap[t[0] as usize], remap[t[1] as usize], remap[t[2] as usize]);
                // A triangle whose corners landed in one cell has no area.
                (a != b && b != c && a != c).then_some([a, b, c])
            })
            .flatten()
            .collect();

        Cluster {
            positions: acc
                .iter()
                .map(|a| {
                    [
                        (a.pos[0] / a.count) as f32,
                        (a.pos[1] / a.count) as f32,
                        (a.pos[2] / a.count) as f32,
                    ]
                })
                .collect(),
            normals: acc
                .iter()
                .map(|a| {
                    let v =
                        glam::Vec3::new(a.normal[0] as f32, a.normal[1] as f32, a.normal[2] as f32);
                    v.normalize_or(glam::Vec3::Y).to_array()
                })
                .collect(),
            uvs: acc
                .iter()
                .map(|a| [(a.uv[0] / a.count) as f32, (a.uv[1] / a.count) as f32])
                .collect(),
            indices,
        }
    }

    /// Scale so the mesh stands `metres` tall, and drop it onto y = 0.
    ///
    /// Downloaded models arrive in whatever units and origin the author used;
    /// a scatter tool that plants a 40 m boulder because the exporter worked in
    /// centimetres is useless. Normalising on import means the per-species
    /// scale range in the UI is in real metres.
    pub fn normalize_height(&mut self, metres: f32) {
        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
        for p in &self.positions {
            lo = lo.min(p[1]);
            hi = hi.max(p[1]);
        }
        let height = (hi - lo).max(1e-6);
        let k = metres / height;
        for p in &mut self.positions {
            p[0] *= k;
            p[1] = (p[1] - lo) * k;
            p[2] *= k;
        }
    }
}

impl MeshData {
    /// Rasterise a preview of this mesh, RGBA, `size` square.
    ///
    /// A palette of names tells you nothing about a scanned asset -- "shrub_03"
    /// and "shrub_sorrel_01" are indistinguishable until you see them. This is
    /// a small software rasteriser rather than a render-to-texture pass because
    /// it needs no GPU state, runs at load, and caches with the mesh.
    ///
    /// Three-quarter view, because an elevation of a tree is a green rectangle.
    pub fn thumbnail(&self, size: u32) -> Vec<u8> {
        let n = size as usize;
        let mut rgba = vec![0u8; n * n * 4];
        let mut depth = vec![f32::MAX; n * n];
        if self.positions.is_empty() {
            return rgba;
        }

        // Turn the model a little off-axis and tip it forward.
        let rot = glam::Mat3::from_rotation_y(0.6) * glam::Mat3::from_rotation_x(-0.25);
        let view: Vec<glam::Vec3> =
            self.positions.iter().map(|p| rot * glam::Vec3::from(*p)).collect();

        let (mut lo, mut hi) = (glam::Vec3::splat(f32::MAX), glam::Vec3::splat(f32::MIN));
        for v in &view {
            lo = lo.min(*v);
            hi = hi.max(*v);
        }
        let span = (hi - lo).max_element().max(1e-6);
        let margin = size as f32 * 0.08;
        let scale = (size as f32 - margin * 2.0) / span;
        let centre = (lo + hi) * 0.5;

        let light = glam::Vec3::new(0.42, 0.72, 0.55).normalize();
        let to_px = |v: glam::Vec3| {
            (
                (v.x - centre.x) * scale + size as f32 * 0.5,
                // Screen y grows downward.
                size as f32 * 0.5 - (v.y - centre.y) * scale,
            )
        };

        for tri in self.indices.chunks_exact(3) {
            let (ia, ib, ic) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
            let (va, vb, vc) = (view[ia], view[ib], view[ic]);
            let (pa, pb, pc) = (to_px(va), to_px(vb), to_px(vc));

            let area = (pb.0 - pa.0) * (pc.1 - pa.1) - (pc.0 - pa.0) * (pb.1 - pa.1);
            if area.abs() < 1e-6 {
                continue;
            }
            // Faces pointing away are still drawn: cut-out foliage is planes,
            // and back-face culling a leaf card removes half the plant.
            let normal = (vb - va).cross(vc - va).normalize_or(glam::Vec3::Y);
            let lambert = normal.dot(light).abs().clamp(0.0, 1.0) * 0.75 + 0.25;

            let x0 = pa.0.min(pb.0).min(pc.0).floor().max(0.0) as usize;
            let x1 = (pa.0.max(pb.0).max(pc.0).ceil() as usize).min(n - 1);
            let y0 = pa.1.min(pb.1).min(pc.1).floor().max(0.0) as usize;
            let y1 = (pa.1.max(pb.1).max(pc.1).ceil() as usize).min(n - 1);

            for y in y0..=y1 {
                for x in x0..=x1 {
                    let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
                    let w0 = ((pb.0 - pa.0) * (py - pa.1) - (px - pa.0) * (pb.1 - pa.1)) / area;
                    let w1 = ((px - pa.0) * (pc.1 - pa.1) - (pc.0 - pa.0) * (py - pa.1)) / area;
                    let w2 = 1.0 - w0 - w1;
                    if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                        continue;
                    }
                    // Barycentrics here are (w2, w1, w0) for (a, b, c).
                    let z = va.z * w2 + vb.z * w1 + vc.z * w0;
                    let idx = y * n + x;
                    if z >= depth[idx] {
                        continue;
                    }

                    let mut rgb = self.base_color;
                    if let Some(tex) = &self.albedo {
                        let uv = [
                            self.uvs.get(ia).map(|u| u[0]).unwrap_or(0.0) * w2
                                + self.uvs.get(ib).map(|u| u[0]).unwrap_or(0.0) * w1
                                + self.uvs.get(ic).map(|u| u[0]).unwrap_or(0.0) * w0,
                            self.uvs.get(ia).map(|u| u[1]).unwrap_or(0.0) * w2
                                + self.uvs.get(ib).map(|u| u[1]).unwrap_or(0.0) * w1
                                + self.uvs.get(ic).map(|u| u[1]).unwrap_or(0.0) * w0,
                        ];
                        let tx = ((uv[0].rem_euclid(1.0)) * tex.width as f32) as u32;
                        let ty = ((uv[1].rem_euclid(1.0)) * tex.height as f32) as u32;
                        let o = ((ty.min(tex.height - 1) * tex.width + tx.min(tex.width - 1)) * 4)
                            as usize;
                        if let Some(p) = tex.rgba.get(o..o + 4) {
                            if let Some(cut) = self.alpha_cutoff {
                                if (p[3] as f32 / 255.0) < cut {
                                    continue;
                                }
                            }
                            // The map is sRGB and so is the output, so no
                            // conversion is needed either way.
                            rgb = [p[0] as f32 / 255.0, p[1] as f32 / 255.0, p[2] as f32 / 255.0];
                        }
                    } else {
                        // Base colour is linear; the thumbnail is sRGB.
                        for c in rgb.iter_mut() {
                            *c = c.max(0.0).powf(1.0 / 2.2);
                        }
                    }

                    depth[idx] = z;
                    for c in 0..3 {
                        rgba[idx * 4 + c] = ((rgb[c] * lambert).clamp(0.0, 1.0) * 255.0) as u8;
                    }
                    rgba[idx * 4 + 3] = 255;
                }
            }
        }
        rgba
    }
}

// ---------------------------------------------------------------------------
// glTF
// ---------------------------------------------------------------------------

/// Model files under `dir`, sorted so palette order is stable between runs.
/// Both loose files and one-folder-per-model layouts work.
pub fn discover(dir: &Path) -> Vec<PathBuf> {
    fn is_model(p: &Path) -> bool {
        p.extension()
            .is_some_and(|x| x.eq_ignore_ascii_case("gltf") || x.eq_ignore_ascii_case("glb"))
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut found = Vec::new();
    for e in entries.flatten() {
        let path = e.path();
        if is_model(&path) {
            found.push(path);
        } else if path.is_dir() {
            let hidden =
                path.file_name().and_then(|s| s.to_str()).is_some_and(|s| s.starts_with('.'));
            if hidden {
                continue;
            }
            if let Ok(inner) = std::fs::read_dir(&path) {
                let mut models: Vec<PathBuf> =
                    inner.flatten().map(|e| e.path()).filter(|p| is_model(p)).collect();
                models.sort();
                found.extend(models.into_iter().take(1));
            }
        }
    }
    found.sort();
    found
}

/// Load every primitive in the file, flattened into one mesh with node
/// transforms applied.
///
/// Flattening is the right call here: a scatter instance is one transform, so a
/// tree that arrives as trunk-plus-canopy nodes has to become a single buffer
/// or every instance would cost several draws.
pub fn load_gltf(path: &Path) -> Result<MeshData> {
    let (doc, buffers, images) =
        gltf::import(path).with_context(|| format!("reading {}", path.display()))?;

    let mut out = MeshData { base_color: [0.42, 0.44, 0.40], ..Default::default() };
    let mut got_color = false;

    for scene in doc.scenes() {
        for node in scene.nodes() {
            visit(&node, glam::Mat4::IDENTITY, &buffers, &images, &mut out, &mut got_color);
        }
    }

    if out.positions.is_empty() {
        anyhow::bail!("{}: no geometry", path.display());
    }
    if out.normals.len() != out.positions.len() {
        out.normals = derive_normals(&out.positions, &out.indices);
    }
    Ok(out)
}

fn visit(
    node: &gltf::Node,
    parent: glam::Mat4,
    buffers: &[gltf::buffer::Data],
    images: &[gltf::image::Data],
    out: &mut MeshData,
    got_color: &mut bool,
) {
    let local = glam::Mat4::from_cols_array_2d(&node.transform().matrix());
    let world = parent * local;
    // Normals transform by the inverse transpose; for the non-uniform scales
    // exporters routinely bake in, using the matrix directly shears them.
    let normal_mat = glam::Mat3::from_mat4(world).inverse().transpose();

    if let Some(mesh) = node.mesh() {
        for prim in mesh.primitives() {
            if prim.mode() != gltf::mesh::Mode::Triangles {
                continue;
            }
            let reader = prim.reader(|b| buffers.get(b.index()).map(|d| &d.0[..]));
            let Some(positions) = reader.read_positions() else { continue };

            let base = out.positions.len() as u32;
            let positions: Vec<[f32; 3]> = positions.collect();
            let count = positions.len();

            for p in positions {
                let v = world.transform_point3(glam::Vec3::from(p));
                out.positions.push(v.to_array());
            }
            match reader.read_normals() {
                Some(ns) => {
                    for n in ns {
                        let v = (normal_mat * glam::Vec3::from(n)).normalize_or_zero();
                        out.normals.push(v.to_array());
                    }
                }
                None => out.normals.extend(std::iter::repeat_n([0.0, 1.0, 0.0], count)),
            }
            match reader.read_tex_coords(0) {
                Some(uvs) => out.uvs.extend(uvs.into_f32()),
                None => out.uvs.extend(std::iter::repeat_n([0.0, 0.0], count)),
            }
            match reader.read_indices() {
                Some(idx) => out.indices.extend(idx.into_u32().map(|i| i + base)),
                // A primitive with no index buffer is a plain triangle list.
                None => out.indices.extend((0..count as u32).map(|i| i + base)),
            }

            if !*got_color {
                let mat = prim.material();
                let pbr = mat.pbr_metallic_roughness();
                let c = pbr.base_color_factor();
                out.base_color = [c[0], c[1], c[2]];
                out.double_sided = mat.double_sided();
                out.alpha_cutoff = match mat.alpha_mode() {
                    // BLEND is treated as a mask too: sorting transparent
                    // foliage per instance is not worth it, and scanned plants
                    // are authored as cut-outs regardless of the declared mode.
                    gltf::material::AlphaMode::Opaque => None,
                    _ => Some(mat.alpha_cutoff().unwrap_or(0.5)),
                };
                if let Some(tex) = pbr.base_color_texture() {
                    out.albedo = decode_image(images.get(tex.texture().source().index()));
                    // A cut-out mask is useless without somewhere to read it
                    // from, and a scanned plant whose material forgot to say
                    // MASK still needs one.
                    if out.alpha_cutoff.is_none()
                        && out.albedo.as_ref().is_some_and(has_transparency)
                    {
                        out.alpha_cutoff = Some(0.5);
                    }
                }
                *got_color = true;
            }
        }
    }

    for child in node.children() {
        visit(&child, world, buffers, images, out, got_color);
    }
}

/// Convert a decoded glTF image to RGBA8.
fn decode_image(data: Option<&gltf::image::Data>) -> Option<Texture> {
    use gltf::image::Format;
    let d = data?;
    let n = (d.width * d.height) as usize;
    let rgba = match d.format {
        Format::R8G8B8A8 => d.pixels.clone(),
        Format::R8G8B8 => {
            let mut v = Vec::with_capacity(n * 4);
            for p in d.pixels.chunks_exact(3) {
                v.extend_from_slice(&[p[0], p[1], p[2], 255]);
            }
            v
        }
        Format::R8 => d.pixels.iter().flat_map(|&g| [g, g, g, 255]).collect(),
        Format::R8G8 => {
            let mut v = Vec::with_capacity(n * 4);
            for p in d.pixels.chunks_exact(2) {
                v.extend_from_slice(&[p[0], p[0], p[0], p[1]]);
            }
            v
        }
        other => {
            log::warn!("unsupported texture format {other:?}");
            return None;
        }
    };
    (rgba.len() == n * 4).then_some(Texture { width: d.width, height: d.height, rgba })
}

fn has_transparency(t: &Texture) -> bool {
    t.rgba.chunks_exact(4).any(|p| p[3] < 250)
}

/// Area-weighted vertex normals, for files that ship none.
fn derive_normals(positions: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut acc = vec![glam::Vec3::ZERO; positions.len()];
    for tri in indices.chunks_exact(3) {
        let (a, b, c) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let (pa, pb, pc) = (
            glam::Vec3::from(positions[a]),
            glam::Vec3::from(positions[b]),
            glam::Vec3::from(positions[c]),
        );
        // Un-normalised cross product is proportional to triangle area, which
        // is the weighting you want -- big faces should dominate.
        let n = (pb - pa).cross(pc - pa);
        acc[a] += n;
        acc[b] += n;
        acc[c] += n;
    }
    acc.into_iter().map(|n| n.normalize_or(glam::Vec3::Y).to_array()).collect()
}

// ---------------------------------------------------------------------------
// Generated species
// ---------------------------------------------------------------------------

/// Built-in shapes, so the scatter tool has something to place before any
/// model has been downloaded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Builtin {
    PineTree,
    BroadleafTree,
    Rock,
    Bush,
}

impl Builtin {
    pub const ALL: [Builtin; 4] =
        [Builtin::PineTree, Builtin::BroadleafTree, Builtin::Rock, Builtin::Bush];

    pub fn name(self) -> &'static str {
        match self {
            Builtin::PineTree => "Pine",
            Builtin::BroadleafTree => "Broadleaf",
            Builtin::Rock => "Rock",
            Builtin::Bush => "Bush",
        }
    }

    /// Default height in metres, used as the species' base scale.
    pub fn height_m(self) -> f32 {
        match self {
            Builtin::PineTree => 14.0,
            Builtin::BroadleafTree => 11.0,
            Builtin::Rock => 2.2,
            Builtin::Bush => 1.4,
        }
    }

    pub fn build(self) -> MeshData {
        match self {
            Builtin::PineTree => pine(),
            Builtin::BroadleafTree => broadleaf(),
            Builtin::Rock => rock(),
            Builtin::Bush => bush(),
        }
    }
}

/// Append a cone with its apex up, base at `y`.
fn cone(m: &mut MeshData, y: f32, radius: f32, height: f32, segments: u32, color_hint: f32) {
    let _ = color_hint;
    let base = m.positions.len() as u32;
    let apex = [0.0, y + height, 0.0];
    m.positions.push(apex);
    m.normals.push([0.0, 1.0, 0.0]);
    m.uvs.push([0.5, 1.0]);

    for i in 0..segments {
        let a = i as f32 / segments as f32 * std::f32::consts::TAU;
        let (s, c) = a.sin_cos();
        m.positions.push([c * radius, y, s * radius]);
        // Side normal: outward and tilted up by the cone's slope.
        let n = glam::Vec3::new(c * height, radius, s * height).normalize();
        m.normals.push(n.to_array());
        m.uvs.push([i as f32 / segments as f32, 0.0]);
    }
    for i in 0..segments {
        let a = base + 1 + i;
        let b = base + 1 + (i + 1) % segments;
        m.indices.extend([base, b, a]);
    }
    // Underside, so the cone is closed when seen from below on a slope.
    let centre = m.positions.len() as u32;
    m.positions.push([0.0, y, 0.0]);
    m.normals.push([0.0, -1.0, 0.0]);
    m.uvs.push([0.5, 0.5]);
    for i in 0..segments {
        let a = base + 1 + i;
        let b = base + 1 + (i + 1) % segments;
        m.indices.extend([centre, a, b]);
    }
}

fn trunk(m: &mut MeshData, radius: f32, height: f32, segments: u32) {
    let base = m.positions.len() as u32;
    for i in 0..segments {
        let a = i as f32 / segments as f32 * std::f32::consts::TAU;
        let (s, c) = a.sin_cos();
        // Tapered: a cylinder reads as a pipe, a taper reads as a trunk.
        let u = i as f32 / segments as f32;
        m.positions.push([c * radius, 0.0, s * radius]);
        m.normals.push([c, 0.0, s]);
        m.uvs.push([u, 0.0]);
        m.positions.push([c * radius * 0.62, height, s * radius * 0.62]);
        m.normals.push([c, 0.0, s]);
        m.uvs.push([u, 1.0]);
    }
    for i in 0..segments {
        let a = base + i * 2;
        let b = base + ((i + 1) % segments) * 2;
        m.indices.extend([a, b, a + 1, b, b + 1, a + 1]);
    }
}

/// Low-poly blob: a subdivided octahedron pushed around by a cheap hash, so
/// every rock is lumpy without being a sphere.
fn blob(m: &mut MeshData, radius: f32, squash: f32, seed: u32) {
    // Deliberately coarse. These are drawn tens of thousands of times, so a
    // triangle here costs more than a triangle anywhere else in the renderer.
    const RINGS: u32 = 4;
    const SEGS: u32 = 7;
    let base = m.positions.len() as u32;
    let jitter = |i: u32, j: u32| {
        let h = (i.wrapping_mul(73_856_093) ^ j.wrapping_mul(19_349_663) ^ seed)
            .wrapping_mul(83_492_791);
        0.76 + ((h >> 8) & 0xFF) as f32 / 255.0 * 0.48
    };

    for r in 0..=RINGS {
        let phi = r as f32 / RINGS as f32 * std::f32::consts::PI;
        for s in 0..SEGS {
            let theta = s as f32 / SEGS as f32 * std::f32::consts::TAU;
            let k = jitter(r, s);
            let p = glam::Vec3::new(
                phi.sin() * theta.cos() * radius * k,
                phi.cos() * radius * squash * k,
                phi.sin() * theta.sin() * radius * k,
            );
            m.positions.push([p.x, p.y + radius * squash, p.z]);
            m.normals.push(p.normalize_or(glam::Vec3::Y).to_array());
            m.uvs.push([s as f32 / SEGS as f32, r as f32 / RINGS as f32]);
        }
    }
    for r in 0..RINGS {
        for s in 0..SEGS {
            let a = base + r * SEGS + s;
            let b = base + r * SEGS + (s + 1) % SEGS;
            let c = base + (r + 1) * SEGS + s;
            let d = base + (r + 1) * SEGS + (s + 1) % SEGS;
            m.indices.extend([a, c, b, b, c, d]);
        }
    }
}

fn pine() -> MeshData {
    let mut m = MeshData { base_color: [0.055, 0.105, 0.052], ..Default::default() };
    trunk(&mut m, 0.34, 5.0, 7);
    cone(&mut m, 3.2, 2.5, 5.0, 9, 0.0);
    cone(&mut m, 6.0, 1.9, 4.6, 9, 0.0);
    cone(&mut m, 8.8, 1.2, 4.0, 9, 0.0);
    m
}

fn broadleaf() -> MeshData {
    let mut m = MeshData { base_color: [0.085, 0.135, 0.058], ..Default::default() };
    trunk(&mut m, 0.42, 5.4, 7);
    blob(&mut m, 3.1, 0.85, 11);
    // Lift the canopy onto the trunk.
    let lift = 4.6;
    for p in m.positions.iter_mut().skip(14) {
        p[1] += lift;
    }
    m
}

fn rock() -> MeshData {
    let mut m = MeshData { base_color: [0.135, 0.132, 0.126], ..Default::default() };
    blob(&mut m, 1.1, 0.72, 4);
    m
}

fn bush() -> MeshData {
    let mut m = MeshData { base_color: [0.072, 0.108, 0.048], ..Default::default() };
    blob(&mut m, 0.7, 0.8, 7);
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_species_are_well_formed() {
        for b in Builtin::ALL {
            let m = b.build();
            assert!(m.triangle_count() > 0, "{} has no triangles", b.name());
            assert_eq!(m.positions.len(), m.normals.len(), "{}", b.name());
            // The vertex layout is shared with imported meshes, so a generated
            // one missing UVs would upload garbage.
            assert_eq!(m.positions.len(), m.uvs.len(), "{} is missing UVs", b.name());
            assert!(
                m.indices.iter().all(|i| (*i as usize) < m.positions.len()),
                "{} indexes past its vertices",
                b.name()
            );
            assert!(
                m.normals.iter().all(|n| {
                    let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                    (l - 1.0).abs() < 1e-3
                }),
                "{} has unnormalised normals",
                b.name()
            );
        }
    }

    #[test]
    fn normalising_height_puts_the_mesh_on_the_ground() {
        let mut m = Builtin::PineTree.build();
        m.normalize_height(20.0);
        let lo = m.positions.iter().map(|p| p[1]).fold(f32::MAX, f32::min);
        let hi = m.positions.iter().map(|p| p[1]).fold(f32::MIN, f32::max);
        // Scatter places instances by their base, so the mesh has to start at
        // zero or every tree floats or sinks by half its height.
        assert!(lo.abs() < 1e-3, "base should sit on y=0, got {lo}");
        assert!((hi - 20.0).abs() < 1e-3, "height should be 20 m, got {hi}");
    }

    #[test]
    fn derived_normals_face_outward() {
        // A single upward triangle must produce upward normals.
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]];
        let normals = derive_normals(&positions, &[0, 1, 2]);
        assert!(normals.iter().all(|n| n[1] > 0.9), "{normals:?}");
    }

    #[test]
    fn decimation_reduces_and_stays_valid() {
        let mut m = Builtin::BroadleafTree.build();
        // Subdivide-free stand-in for a scanned asset: repeat the mesh so it
        // is dense enough to have something to remove.
        let base = m.clone();
        for _ in 0..3 {
            let off = m.positions.len() as u32;
            m.positions.extend(base.positions.iter().map(|p| [p[0] + 0.01, p[1], p[2]]));
            m.normals.extend(base.normals.iter().copied());
            m.uvs.extend(base.uvs.iter().copied());
            m.indices.extend(base.indices.iter().map(|i| i + off));
        }
        let before = m.triangle_count();
        let target = before / 4;
        m.decimate(target);
        let after = m.triangle_count();
        assert!(after <= target, "must meet the target: {before} -> {after} (target {target})");
        // And must not throw away far more than asked. Overshooting is how a
        // 100k-triangle scan turns into 300 triangles of mush.
        assert!(after * 4 > target, "over-decimated: {before} -> {after} (target {target})");
        assert_eq!(m.positions.len(), m.normals.len());
        assert_eq!(m.positions.len(), m.uvs.len());
        assert!(m.indices.iter().all(|i| (*i as usize) < m.positions.len()));
        assert!(m.positions.iter().all(|p| p.iter().all(|c| c.is_finite())));
    }

    #[test]
    fn decimation_leaves_a_small_mesh_alone() {
        let m = Builtin::Rock.build();
        let mut d = m.clone();
        d.decimate(1_000_000);
        assert_eq!(d.triangle_count(), m.triangle_count());
    }

    #[test]
    fn thumbnail_draws_something() {
        for b in Builtin::ALL {
            let m = b.build();
            let img = m.thumbnail(48);
            assert_eq!(img.len(), 48 * 48 * 4);
            let covered = img.chunks_exact(4).filter(|p| p[3] > 0).count();
            // A blank preview is worse than no preview: it looks like a broken
            // asset rather than a missing feature.
            assert!(covered > 200, "{} rendered only {covered} pixels", b.name());
            assert!(covered < 48 * 48, "{} filled the frame; the fit is wrong", b.name());
        }
    }

    #[test]
    fn bounding_radius_covers_every_vertex() {
        let m = Builtin::BroadleafTree.build();
        let r = m.bounding_radius();
        assert!(
            m.positions
                .iter()
                .all(|p| { (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt() <= r + 1e-4 })
        );
    }
}
