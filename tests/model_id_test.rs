//! Model-id reporting gate: the server must report the model card's `base_model:` as the
//! public id, not the lab directory name. Before the fix, /v1/models and every chat response
//! carried `"model": "3.8-27b-nvfp4-full-all"` (an internal path fragment no client can
//! resolve). Run: `cargo test --test model_id_test`.

use gb10_inference::server::model_id_from_dir;
use std::io::Write;

fn tmp_model_dir(card: Option<&str>) -> String {
    let d = std::env::temp_dir().join(format!(
        "mid_test_{}_{}",
        std::process::id(),
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
    ));
    std::fs::create_dir_all(&d).unwrap();
    if let Some(c) = card {
        let mut f = std::fs::File::create(d.join("README.md")).unwrap();
        f.write_all(c.as_bytes()).unwrap();
    }
    d.to_string_lossy().to_string()
}

#[test]
fn reports_base_model_from_card() {
    let dir = tmp_model_dir(Some("---\nlicense: apache-2.0\nbase_model: Qwen/Qwen3.8-27B\ntags:\n- nvfp4\n---\n# card\n"));
    assert_eq!(model_id_from_dir(&dir), "Qwen/Qwen3.8-27B");
}

#[test]
fn quoted_base_model_stripped() {
    let dir = tmp_model_dir(Some("---\nbase_model: \"Qwen/Qwen3.5-2B\"\n---\n"));
    assert_eq!(model_id_from_dir(&dir), "Qwen/Qwen3.5-2B");
}

#[test]
fn falls_back_to_dirname_without_line() {
    let d = tmp_model_dir(Some("---\nlicense: apache-2.0\n---\n# no base_model line\n"));
    let expect = std::path::Path::new(&d).file_name().unwrap().to_string_lossy().to_string();
    assert_eq!(model_id_from_dir(&d), expect);
}

#[test]
fn falls_back_to_dirname_without_card() {
    let d = tmp_model_dir(None);
    let expect = std::path::Path::new(&d).file_name().unwrap().to_string_lossy().to_string();
    assert_eq!(model_id_from_dir(&d), expect);
}

#[test]
fn empty_value_falls_back() {
    let d = tmp_model_dir(Some("---\nbase_model:\n---\n"));
    let expect = std::path::Path::new(&d).file_name().unwrap().to_string_lossy().to_string();
    assert_eq!(model_id_from_dir(&d), expect);
}
