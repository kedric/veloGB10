//! HOTFIX gate (2026-08-27): malformed tool-call tag + streaming/non-streaming divergence.
//!
//! User report on build 0.4.2 (dev b803b41): the model sometimes emits the function tag with
//! the `<` missing — `<tool_call>\nfunction=get_weather>` (live capture: 2 of 38 sampled tool
//! requests, temp 1.0) — and the engine then diverged by mode: the STREAMING serializer
//! silently dropped the whole block's text (it had been held back and nothing parsed), while
//! the NON-STREAMING serializer leaked the raw block into `content` (recoverable). The fix is
//! ONE shared decision (`tools::finalize_parsed`) for both modes, a parser repair
//! (`tools::find_function_tag`) that reads the bare `function=NAME>` form, and a
//! no-silent-drop rule for held-back text (`tools::held_back_remainder`).
//!
//! Expected wire form is the MODEL's own chat_template.jinja (NOT guessed):
//!   <tool_call>\n<function=NAME>\n<parameter=K>\nvalue\n</parameter>\n</function>\n</tool_call>
//! and hy_v3's `<tool_calls:opensource>` / `<tool_call:opensource>NAME<tool_sep:opensource>…`.
//! Run: `cargo test --test toolcall_serial_test`.

use gb10_inference::tools;
use serde_json::Value;

/// The EXACT malformed raw text captured live from the model (seed-6 sweep, 2026-08-27):
/// `<tool_call>` intact, `<` of `<function=` missing. This is the case that tripped.
const RAW_MALFORMED_CAPTURED: &str = "<tool_call>\nfunction=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=units>\nc\n</parameter>\n</function>\n</tool_call>";

const RAW_WELLFORMED: &str = "<tool_call>\n<function=get_weather>\n<parameter=city>\nBerlin\n</parameter>\n</function>\n</tool_call>";

const RAW_MULTI: &str = "I'll check both cities.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n</function>\n</tool_call>";

const RAW_NO_ARGS: &str = "<tool_call>\n<function=list_cities>\n</function>\n</tool_call>";

const RAW_UNICODE: &str = "<tool_call>\n<function=get_weather>\n<parameter=city>\nZürich 🇨🇭 北çå\n</parameter>\n<parameter=note>\ntempérature — très élevée\n</parameter>\n</function>\n</tool_call>";

const RAW_HY3: &str = "Checking.<tool_calls:opensource>\n<tool_call:opensource>get_weather<tool_sep:opensource>\n<arg_key:opensource>city</arg_key:opensource>\n<arg_value:opensource>\nParis\n</arg_value:opensource>\n</tool_call:opensource>\n</tool_calls:opensource>";

/// A truncated block (no close tag) — parse drops it; BOTH modes must then surface the raw
/// text (non-streaming always did; streaming via held_back_remainder).
const RAW_TRUNCATED: &str = "Let me check.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis";

fn schema() -> Option<Vec<Value>> {
    let s = std::fs::read_to_string(
        std::env::var("GB10_TEST_MODEL_DIR")
            .map(std::path::PathBuf::from)
            .map(|p| p.join("tools_fixture.json"))
            .unwrap_or_else(|_| std::path::PathBuf::from("/nonexistent")),
    )
    .ok();
    let _ = s; // schema comes from the request, not disk — kept None so coercion stays string-typed
    None
}

/// Strict wire well-formedness: the emitted call must round-trip as the OpenAI tool_calls
/// shape AND its `arguments` must be a JSON object (the harness feeds it to a real function).
fn assert_wire_wellformed(calls: &[gb10_inference::tokenizer::ToolCall]) {
    // Strict only when a call IS emitted; zero calls is a legal outcome (dropped/truncated
    // block) covered by the no-silent-drop assertions instead.
    if calls.is_empty() { return; }
    for c in calls {
        assert_eq!(c.kind, "function", "type must be \"function\" on the wire");
        assert!(!c.id.is_empty(), "call id must be non-empty");
        assert!(!c.function.name.is_empty(), "function name must be non-empty");
        assert!(
            !c.function.name.contains('<') && !c.function.name.contains('>') && !c.function.name.contains('\n'),
            "name must be a bare identifier, not tag fragments: {:?}",
            c.function.name
        );
        let args: Value = serde_json::from_str(&c.function.arguments)
            .unwrap_or_else(|e| panic!("arguments must be a JSON object, got {:?}: {}", c.function.arguments, e));
        assert!(args.is_object(), "arguments must deserialize to a JSON object");
    }
}

/// The MODE-AGREEMENT invariant, computed exactly the way the two server paths compute it:
///  - non-streaming: content/tool_calls = tools::finalize(raw)
///  - streaming:     prose streams incrementally up to the hold-back (modeled by the prefix
///                   the stream would have emitted — everything before the first marker),
///                   then the same finalize decides the tool_calls, and anything held back
///                   while nothing parsed is surfaced via held_back_remainder.
/// Asserts: same tool_calls (name + arguments JSON), same finish_reason, and the FULL text
/// content the client sees agrees between modes.
fn assert_modes_agree(raw: &str) {
    let tools = schema();
    let finish_in = "stop";

    // --- non-streaming mode ---
    let (sync_content, sync_calls, sync_finish) = tools::finalize(raw, tools.as_deref(), finish_in);

    // --- streaming mode (model of server.rs's stream) ---
    // The stream emits prose up to the first tool-call marker; everything after is held back.
    let hold = raw.find("<tool_call").unwrap_or(raw.len());
    let mut streamed_text = raw[..hold].to_string();
    let content_emitted = hold;
    let parsed = tools::parse(raw, tools.as_deref());
    let (_, stream_calls, stream_finish) = tools::finalize_parsed(raw, parsed, finish_in);
    if stream_calls.is_empty() {
        if let Some(held) = tools::held_back_remainder(raw, content_emitted) {
            streamed_text.push_str(held);
        }
    }

    // 1. Same tool-call content (name + arguments JSON), not just same count.
    assert_eq!(sync_calls.len(), stream_calls.len(), "call count diverged for {raw:?}");
    for (s, j) in stream_calls.iter().zip(sync_calls.iter()) {
        assert_eq!(s.function.name, j.function.name, "call name diverged for {raw:?}");
        assert_eq!(s.function.arguments, j.function.arguments, "call arguments diverged for {raw:?}");
    }
    // 2. Same finish_reason.
    assert_eq!(sync_finish, stream_finish, "finish_reason diverged for {raw:?}");
    // 3. No dropped text: the text the streaming client assembles equals what non-streaming
    //    returns as content (prose-only when calls parsed; the raw answer text when not).
    let sync_text = sync_content.clone().unwrap_or_default();
    assert_eq!(
        streamed_text.trim(),
        sync_text.trim(),
        "streaming text != non-streaming content for {raw:?}"
    );
    // 4. Wire tag well-formed on whatever calls were emitted.
    assert_wire_wellformed(&sync_calls);
}

#[test]
fn failing_case_malformed_tag_parses_and_both_modes_agree() {
    // BEFORE the fix: parse failed -> streaming DROPPED the block (no calls, no content),
    // non-streaming LEAKED the raw block into content. Mode agreement was impossible.
    let tools = schema();
    let parsed = tools::parse(RAW_MALFORMED_CAPTURED, tools.as_deref());
    assert_eq!(parsed.tool_calls.len(), 1, "the malformed (bare function=) form must parse");
    assert_eq!(parsed.tool_calls[0].function.name, "get_weather");
    let args: Value = serde_json::from_str(&parsed.tool_calls[0].function.arguments).unwrap();
    assert_eq!(args["city"], "Paris");
    assert_eq!(args["units"], "c");
    assert_modes_agree(RAW_MALFORMED_CAPTURED);
}

#[test]
fn known_passing_wellformed_call() {
    let tools = schema();
    let parsed = tools::parse(RAW_WELLFORMED, tools.as_deref());
    assert_eq!(parsed.tool_calls.len(), 1);
    assert_eq!(parsed.tool_calls[0].function.name, "get_weather");
    let args: Value = serde_json::from_str(&parsed.tool_calls[0].function.arguments).unwrap();
    assert_eq!(args["city"], "Berlin");
    assert_modes_agree(RAW_WELLFORMED);
}

#[test]
fn multi_tool_call() {
    let tools = schema();
    let parsed = tools::parse(RAW_MULTI, tools.as_deref());
    assert_eq!(parsed.tool_calls.len(), 2, "both calls must parse");
    assert_eq!(parsed.tool_calls[0].function.name, "get_weather");
    assert_eq!(parsed.tool_calls[1].function.name, "get_weather");
    let a0: Value = serde_json::from_str(&parsed.tool_calls[0].function.arguments).unwrap();
    let a1: Value = serde_json::from_str(&parsed.tool_calls[1].function.arguments).unwrap();
    assert_eq!(a0["city"], "Paris");
    assert_eq!(a1["city"], "Tokyo");
    assert_eq!(parsed.content, "I'll check both cities.");
    assert_modes_agree(RAW_MULTI);
}

#[test]
fn call_with_no_arguments() {
    let tools = schema();
    let parsed = tools::parse(RAW_NO_ARGS, tools.as_deref());
    assert_eq!(parsed.tool_calls.len(), 1);
    assert_eq!(parsed.tool_calls[0].function.name, "list_cities");
    let args: Value = serde_json::from_str(&parsed.tool_calls[0].function.arguments).unwrap();
    assert_eq!(args.as_object().map(|m| m.len()), Some(0), "no-arg call must be the empty object");
    assert_modes_agree(RAW_NO_ARGS);
}

#[test]
fn unicode_in_arguments() {
    let tools = schema();
    let parsed = tools::parse(RAW_UNICODE, tools.as_deref());
    assert_eq!(parsed.tool_calls.len(), 1);
    let args: Value = serde_json::from_str(&parsed.tool_calls[0].function.arguments).unwrap();
    assert_eq!(args["city"], "Zürich 🇨🇭 北çå");
    assert_eq!(args["note"], "température — très élevée");
    assert_modes_agree(RAW_UNICODE);
}

#[test]
fn hy_v3_opensource_form() {
    let tools = schema();
    let parsed = tools::parse(RAW_HY3, tools.as_deref());
    assert_eq!(parsed.tool_calls.len(), 1, "hy_v3 :opensource form must parse");
    assert_eq!(parsed.tool_calls[0].function.name, "get_weather");
    let args: Value = serde_json::from_str(&parsed.tool_calls[0].function.arguments).unwrap();
    assert_eq!(args["city"], "Paris");
    assert_eq!(parsed.content, "Checking.");
    assert_modes_agree(RAW_HY3);
}

#[test]
fn truncated_block_never_silently_dropped() {
    // No close tag -> the call is dropped by the parser (half a call is worse than none), but
    // BOTH modes must surface the raw text: non-streaming returns it as content, streaming
    // via held_back_remainder. This is the no-silent-drop invariant, not a parse assertion.
    let tools = schema();
    let (content, calls, finish) = tools::finalize(RAW_TRUNCATED, tools.as_deref(), "stop");
    assert!(calls.is_empty(), "truncated call must not parse");
    assert_eq!(finish, "stop", "finish passes through when nothing parsed");
    let content = content.expect("raw text must be surfaced, not dropped");
    assert!(content.contains("<tool_call"), "the held-back block text must be recoverable");
    // held_back_remainder must return the buffered span for the streaming mode.
    let hold = RAW_TRUNCATED.find("<tool_call").unwrap();
    assert_eq!(tools::held_back_remainder(RAW_TRUNCATED, hold), Some(&RAW_TRUNCATED[hold..]));
    assert_modes_agree(RAW_TRUNCATED);
}

#[test]
fn malformed_repair_prefers_wellformed_and_rejects_value_text() {
    // The well-formed tag always wins when present ...
    let both = "<tool_call>\n<function=real_fn>\n<parameter=q>\nfunction=decoy>\n</parameter>\n</function>\n</tool_call>";
    let tools = schema();
    let parsed = tools::parse(both, tools.as_deref());
    assert_eq!(parsed.tool_calls.len(), 1);
    assert_eq!(parsed.tool_calls[0].function.name, "real_fn");
    let args: Value = serde_json::from_str(&parsed.tool_calls[0].function.arguments).unwrap();
    assert_eq!(args["q"], "function=decoy>", "the in-value occurrence must stay payload");
    // ... and the bare-form repair only fires at a line boundary, never mid-word.
    let midword = "<tool_call>\nsomefunction=x>\n</tool_call>";
    assert!(tools::parse(midword, tools.as_deref()).tool_calls.is_empty());
}

#[test]
fn no_drop_when_no_tools_at_all() {
    // Ordinary prose (no markers): held_back_remainder must be None and finalize passes the
    // text through untouched — the serialization change must not alter plain text.
    let plain = "The sky is blue because Rayleigh scattering.";
    let tools = schema();
    let (content, calls, finish) = tools::finalize(plain, tools.as_deref(), "stop");
    assert_eq!(calls.len(), 0);
    assert_eq!(content.as_deref(), Some(plain));
    assert_eq!(finish, "stop");
    assert_eq!(tools::held_back_remainder(plain, plain.len()), None);
}
