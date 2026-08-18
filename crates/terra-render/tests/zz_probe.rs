#[test]
fn probe() {
    let d = std::path::Path::new(
        "/Users/johnroshan/Library/Application Support/in.synctric.Terra/projects/New_World/assets/textures/rocky_terrain_03",
    );
    if !d.exists() {
        eprintln!("absent");
        return;
    }
    println!("name      {}", terra_render::texture_set::material_name(d));
    println!("missing   {:?}", terra_render::texture_set::missing_maps(d));
    println!(
        "role      {}",
        terra_render::material::role_label(terra_render::material::role_of(
            &terra_render::texture_set::material_name(d)
        ))
    );
    let s = terra_render::texture_set::load(d, 512).expect("load");
    let rng = |v: &[u8], stride: usize, off: usize| {
        let mut lo = 255u8;
        let mut hi = 0u8;
        for c in v.chunks_exact(stride) {
            lo = lo.min(c[off]);
            hi = hi.max(c[off]);
        }
        (lo, hi)
    };
    println!(
        "normal R  {:?}  G {:?}   (128,128 flat = fallback)",
        rng(&s.normal, 4, 0),
        rng(&s.normal, 4, 1)
    );
    let r =
        (s.roughness.iter().copied().min().unwrap(), s.roughness.iter().copied().max().unwrap());
    let h = (s.height.iter().copied().min().unwrap(), s.height.iter().copied().max().unwrap());
    let a = rng(&s.albedo, 4, 0);
    println!("roughness {r:?}   (200,200 = fallback)");
    println!("height    {h:?}   (128,128 = fallback)");
    println!("albedo R  {a:?}");
    println!("albedo_is_linear {}", s.albedo_is_linear);
}
