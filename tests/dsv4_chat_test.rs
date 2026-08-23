//! Phase 4a GOLDEN GATE: byte-exactness of the DSV4 chat encoder vs the bundle's golden vectors.
//!
//! The bundle ships `encoding/tests/test_input_{1..4}.json` (requests) and `test_output_{1..4}.txt`
//! (the byte-exact expected wire strings produced by `encoding_dsv4.py::encode_messages`). This is
//! the authoritative gate for the chat-format + tokenizer surface: every byte, including the absence
//! of a trailing newline, must match. Run: `cargo test --test dsv4_chat_test`.
//!
//! Case map (mirrors `test_encoding_dsv4.py`):
//!   1 — thinking mode + tools (multi-turn, tool results merged into user; DSML emit + parse)
//!   2 — thinking mode, no tools (drop_thinking strips earlier reasoning)
//!   3 — interleaved thinking + search (developer+tools, latest_reminder, CJK)
//!   4 — quick instruction task (chat mode, action task token)
//!
//! These read the bundle under `/mnt/models/DeepSeek-V4-Flash-DSpark` (READ-ONLY). If the bundle is
//! absent (e.g. a CI box without the model), every case `return`s early so the suite stays green
//! without the ground truth — but on the dev box it is mandatory.

use gb10_inference::dsv4_chat::{self, Json};

const BUNDLE: &str = "/mnt/models/DeepSeek-V4-Flash-DSpark/encoding/tests";

fn bundle_present() -> bool {
    std::path::Path::new(&format!("{BUNDLE}/test_output_1.txt")).exists()
}

/// sha256 of a byte string (hex). Asserting this — not just byte length — catches single-byte drift
/// that a length check would miss (§3: assert the SUCCESS signal, with a concrete fingerprint).
fn sha256_hex(b: &[u8]) -> String {
    // Minimal SHA-256 (no extra dep) — enough for a fingerprint in the assertion message.
    use std::fmt::Write;
    let h = sha256(b);
    let mut s = String::with_capacity(64);
    for w in h { write!(s, "{:08x}", w).unwrap(); }
    s
}
fn sha256(msg: &[u8]) -> [u32; 8] {
    const K: [u32; 64] = [
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
    ];
    let mut h: [u32; 8] = [0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19];
    let mut msg = msg.to_vec();
    let bitlen = (msg.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 { msg.push(0); }
    msg.extend_from_slice(&bitlen.to_be_bytes());
    for chunk in msg.chunks(64) {
        let mut w = [0u32; 64];
        for (i, b) in chunk.chunks(4).enumerate().take(16) {
            w[i] = u32::from_be_bytes([b[0],b[1],b[2],b[3]]);
        }
        for i in 16..64 {
            let s0 = w[i-15].rotate_right(7) ^ w[i-15].rotate_right(18) ^ (w[i-15] >> 3);
            let s1 = w[i-2].rotate_right(17) ^ w[i-2].rotate_right(19) ^ (w[i-2] >> 10);
            w[i] = w[i-16].wrapping_add(s0).wrapping_add(w[i-7]).wrapping_add(s1);
        }
        let (mut a,mut b,mut c,mut d,mut e,mut f,mut g,mut hh) = (h[0],h[1],h[2],h[3],h[4],h[5],h[6],h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh=g; g=f; f=e; e=d.wrapping_add(t1); d=c; c=b; b=a; a=t1.wrapping_add(t2);
        }
        h[0]=h[0].wrapping_add(a); h[1]=h[1].wrapping_add(b); h[2]=h[2].wrapping_add(c); h[3]=h[3].wrapping_add(d);
        h[4]=h[4].wrapping_add(e); h[5]=h[5].wrapping_add(f); h[6]=h[6].wrapping_add(g); h[7]=h[7].wrapping_add(hh);
    }
    h
}

/// Load a case: parse its `test_input_N.json`, attach tools to messages[0] when the input carries a
/// top-level `tools` field (exactly what `test_encoding_dsv4.py::test_case_1` does), and return the
/// messages array + the thinking_mode to use.
fn load_case(n: u8) -> Option<(Vec<Json>, &'static str)> {
    let input = std::fs::read_to_string(format!("{BUNDLE}/test_input_{n}.json")).ok()?;
    let v = dsv4_chat::parse_json(&input).expect("input json");
    let thinking_mode: &'static str = if n == 4 { "chat" } else { "thinking" };
    let messages: Vec<Json> = match &v {
        Json::Array(messages) => messages.clone(),
        obj => {
            // test_input_1 is an object {tools, messages}; the harness attaches tools to messages[0].
            let tools = obj.get("tools").cloned();
            let mut messages = obj.get("messages").and_then(|m| m.as_array()).expect("messages array").to_vec();
            if let Some(t) = tools {
                if let Some(Json::Object(o)) = messages.get_mut(0) {
                    o.push(("tools".into(), t));
                }
            }
            messages
        }
    };
    Some((messages, thinking_mode))
}

/// The per-case byte-exact assertion: same length AND same sha256 AND same bytes.
fn check_byte_exact(n: u8) {
    if !bundle_present() {
        eprintln!("skip case {n}: bundle not present at {BUNDLE}");
        return;
    }
    let (messages, mode) = load_case(n).expect("load case");
    let got = dsv4_chat::encode_messages(&messages, &dsv4_chat::EncodeOptions {
        thinking_mode: mode.into(),
        ..Default::default()
    });
    let gold = std::fs::read(format!("{BUNDLE}/test_output_{n}.txt")).expect("read gold");
    let gold_s = String::from_utf8(gold).expect("gold utf8");

    let gb = gold_s.as_bytes();
    let gb_len = gb.len();
    let got_len = got.as_bytes().len();
    eprintln!("case {n}: got {} bytes, gold {} bytes | got_sha={} gold_sha={}",
              got_len, gb_len, sha256_hex(got.as_bytes()), sha256_hex(gb));

    // Find the first divergence for a crisp failure message.
    if got != gold_s {
        let cp = got_s_first_diff(&got, &gold_s);
        eprintln!("  first diff at byte {}:\n    got:  {:?}\n    gold: {:?}",
                  cp.0, cp.1, cp.2);
    }
    assert_eq!(got_len, gb_len, "case {n}: byte length mismatch (no trailing-newline drift allowed)");
    assert_eq!(got.as_bytes(), gb, "case {n}: byte content mismatch (sha got={} gold={})",
               sha256_hex(got.as_bytes()), sha256_hex(gb));
}

fn got_s_first_diff(a: &str, b: &str) -> (usize, String, String) {
    let ab = a.as_bytes();
    let bb = b.as_bytes();
    let n = ab.len().min(bb.len());
    for i in 0..n {
        if ab[i] != bb[i] {
            let s = i.saturating_sub(20);
            return (i,
                    String::from_utf8_lossy(&ab[s..(i + 20).min(ab.len())]).into_owned(),
                    String::from_utf8_lossy(&bb[s..(i + 20).min(bb.len())]).into_owned());
        }
    }
    (n, String::from_utf8_lossy(&ab[n.min(ab.len())..]).into_owned(),
        String::from_utf8_lossy(&bb[n.min(bb.len())..]).into_owned())
}

#[test]
fn golden_case_1_thinking_with_tools() { check_byte_exact(1); }

#[test]
fn golden_case_2_thinking_drop_thinking() { check_byte_exact(2); }

#[test]
fn golden_case_3_interleaved_search_cjk() { check_byte_exact(3); }

#[test]
fn golden_case_4_action_task_chat_mode() { check_byte_exact(4); }

/// The parser round-trips: re-parse the synthesized assistant turns the way test_encoding_dsv4.py
/// does (marker = `<｜Assistant｜><think>`), asserting the structured fields it checks.
#[test]
fn parse_marker_case1_from_python_test() {
    if !bundle_present() { eprintln!("skip: bundle not present"); return; }
    let (messages, mode) = load_case(1).expect("load case 1");
    let prompt = dsv4_chat::encode_messages(&messages, &dsv4_chat::EncodeOptions {
        thinking_mode: mode.into(), ..Default::default()
    });
    let marker = "<｜Assistant｜><think>";
    // First assistant turn (the tool call) — between the first marker and the next <｜User｜>.
    let first_start = prompt.find(marker).expect("marker") + marker.len();
    let first_end = first_start + prompt[first_start..].find("<｜User｜>").expect("user after first assistant");
    let parsed = dsv4_chat::parse_completion(&prompt[first_start..first_end], "thinking");
    assert_eq!(parsed.reasoning_content, "The user wants to know the weather in Beijing. I should use the get_weather tool.");
    assert_eq!(parsed.content, "");
    assert_eq!(parsed.tool_calls.len(), 1);
    assert_eq!(parsed.tool_calls[0].get("function").unwrap().get("name").unwrap().as_str(), Some("get_weather"));
    let args = parsed.tool_calls[0].get("function").unwrap().get("arguments").unwrap().as_str().unwrap();
    let r = dsv4_chat::parse_json(args).expect("args json");
    assert_eq!(r.get("location").unwrap().as_str(), Some("Beijing"));
    assert_eq!(r.get("unit").unwrap().as_str(), Some("celsius"));

    // Final assistant turn (content, no tool calls) — from the last marker to end.
    let last_start = prompt.rfind(marker).expect("last marker") + marker.len();
    let parsed_final = dsv4_chat::parse_completion(&prompt[last_start..], "thinking");
    assert_eq!(parsed_final.reasoning_content, "Got the weather data. Let me format a nice response.");
    assert!(parsed_final.content.contains("22°C"));
    assert!(parsed_final.tool_calls.is_empty());
}
