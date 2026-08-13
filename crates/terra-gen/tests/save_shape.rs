//! Produces a real world on disk and prints what it looks like. Ignored by
//! default; run with `-- --ignored --nocapture` to inspect the save format.

use terra_core::WorldSize;
use terra_project::{Project, WorldData};

#[test]
#[ignore]
fn dump_saved_world() {
    let root = std::env::temp_dir().join("terra-save-demo/DesertRally");
    let _ = std::fs::remove_dir_all(root.parent().unwrap());

    let size = WorldSize::Small;
    let project = Project::create(&root, "Desert Rally", size, 0x5EED_1234).unwrap();

    let res = size.tier0_res();
    let extent = size.extent_m() as f32;
    let p = project.world.terrain;
    let base: Vec<f32> =
        terra_gen::heightfield::generate(res, extent, &p.rmf).iter().map(|h| h + 256.0).collect();

    // Real erosion, so the saved masks are representative. A constant fill
    // compresses to nothing and would make any size measurement meaningless.
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .unwrap();
    let limits = adapter.limits();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: None,
        required_features: wgpu::Features::empty(),
        required_limits: limits,
        ..Default::default()
    }))
    .unwrap();

    let cell = extent / (res - 1) as f32;
    let sim = terra_gen::erosion::Erosion::new(&device, &queue, res, cell, &p.erosion);
    let run = sim.run(&device, &queue, &base, p.erosion.iterations, |_| {});

    let data = WorldData {
        flow: terra_gen::erosion::Erosion::normalize_flow(&run.flow),
        deposition: terra_gen::erosion::Erosion::deposition_map(&base, &run.height),
        heights: run.height,
    };
    data.save(&project.paths, size).unwrap();
    project.save().unwrap();

    println!("\n=== {} ===", root.display());
    let mut entries = Vec::new();
    fn walk(dir: &std::path::Path, base: &std::path::Path, out: &mut Vec<(String, u64)>) {
        let mut items: Vec<_> = std::fs::read_dir(dir).unwrap().filter_map(|e| e.ok()).collect();
        items.sort_by_key(|e| e.path());
        for e in items {
            let p = e.path();
            let rel = p.strip_prefix(base).unwrap().display().to_string();
            if p.is_dir() {
                out.push((format!("{rel}/"), 0));
                walk(&p, base, out);
            } else {
                out.push((rel, std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0)));
            }
        }
    }
    walk(&root, &root, &mut entries);
    for (name, size) in &entries {
        if *size == 0 {
            println!("  {name}");
        } else if *size < 4096 {
            println!("  {name:<44} {size} B");
        } else {
            println!("  {name:<44} {:.1} MB", *size as f64 / 1_048_576.0);
        }
    }

    println!("\n=== project.ron ===");
    println!("{}", std::fs::read_to_string(project.paths.project_manifest()).unwrap());
    println!("=== world.ron (first 22 lines) ===");
    let w = std::fs::read_to_string(project.paths.world_manifest()).unwrap();
    for line in w.lines().take(22) {
        println!("{line}");
    }
}
