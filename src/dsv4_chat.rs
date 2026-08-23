//! DeepSeek-V4 chat wire-format encoder + DSML tool-calling + output parser.
//!
//! A faithful, byte-exact Rust port of the bundle's `encoding/encoding_dsv4.py` (the ONLY spec;
//! golden vectors live in `encoding/tests/test_output_{1..4}.txt`). The .py is truth — if anything
//! here ever disagrees with it, the .py wins and this is the bug.
//!
//! Why this exists as a standalone module (and NOT the Jinja chat-template path in `tokenizer.rs`):
//! DSV4 ships NO `chat_template.jinja`. Its chat format is defined entirely by `encoding_dsv4.py`,
//! which builds the wire string by hand with literal special-token text (`<｜User｜>`, `｜DSML｜`, …).
//! The tokenizer (HF fast BPE via the `tokenizers` crate) matches those `special:false` added tokens
//! in-text as single ids (verified: `x<｜User｜>y` → `[90, 128803, 91]`), so the recipe is: build the
//! wire STRING byte-exactly here, then `tokenizer.encode(wire, add_special_tokens=false)` (BOS comes
//! from the encoding layer per §E.1, `add_bos_token:false` at the tokenizer level).
//!
//! The one genuinely hard requirement is byte-exact JSON for the tool schemas: the .py round-trips
//! them through `json.dumps(ensure_ascii=False)` (separators `(', ', ': ')`, key INSERTION order
//! preserved, `/` NOT escaped, non-ASCII raw). `serde_json` without `preserve_order` sorts keys and
//! uses `,`/`:` — both wrong. Rather than flip a global serde_json feature (which would change
//! `serde_json::Map` iteration order for every other path), this module ships its own ordered JSON
//! value + parser + Python-compatible emitter, isolated to DSV4. It is verified byte-exact against
//! `json.dumps` on every golden tool schema (see `tests` below).

// =================================================================================================
// Ordered JSON — parser + Python-compatible emitter (== json.dumps(ensure_ascii=False))
// =================================================================================================

/// Ordered JSON value. Object variants preserve key INSERTION order (a `Vec<(String, Json)>`, not a
/// BTreeMap) so `dumps` reproduces `json.dumps` byte-for-byte. Numbers keep their original source
/// text: Python parses `1e3` → float and re-emits `1000.0`, but every real client (and every golden
/// vector) sends already-normalized JSON, so verbatim re-emit is both simplest and byte-exact here.
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    /// Raw number text from the source (e.g. `-1`, `1.5`, `false` is `Bool` not this).
    Num(String),
    Str(String),
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
}

impl Json {
    pub fn as_str(&self) -> Option<&str> {
        if let Json::Str(s) = self { Some(s) } else { None }
    }
    pub fn as_array(&self) -> Option<&[Json]> {
        if let Json::Array(a) = self { Some(a) } else { None }
    }
    pub fn as_bool(&self) -> Option<bool> {
        if let Json::Bool(b) = self { Some(*b) } else { None }
    }
    /// Parse a `Num`'s raw text as u64 (tool-schema `default`, request `max_tokens`, …).
    pub fn as_u64(&self) -> Option<u64> {
        if let Json::Num(s) = self { s.parse().ok() } else { None }
    }
    pub fn as_object(&self) -> Option<&[(String, Json)]> {
        if let Json::Object(o) = self { Some(o) } else { None }
    }
    /// Object field lookup (linear; tool schemas are tiny).
    pub fn get(&self, key: &str) -> Option<&Json> {
        if let Json::Object(o) = self {
            o.iter().find_map(|(k, v)| if k == key { Some(v) } else { None })
        } else { None }
    }
    pub fn is_null(&self) -> bool { matches!(self, Json::Null) }
}

/// Parse error: byte offset + message (enough to localize a malformed request body).
#[derive(Debug)]
pub struct JsonError(pub usize, pub String);

/// Parse a JSON document into ordered `Json`. Accepts exactly what `json.loads` accepts (incl. the
/// `\/` escape, which clients do send and Python silently accepts).
pub fn parse_json(input: &str) -> Result<Json, JsonError> {
    let mut p = JParser { b: input.as_bytes(), i: 0 };
    p.ws();
    let v = p.value()?;
    p.ws();
    if p.i != p.b.len() {
        return Err(JsonError(p.i, format!("trailing data at byte {}", p.i)));
    }
    Ok(v)
}

struct JParser<'a> { b: &'a [u8], i: usize }

impl<'a> JParser<'a> {
    fn err(&self, msg: impl Into<String>) -> JsonError { JsonError(self.i, msg.into()) }
    fn peek(&self) -> Option<u8> { self.b.get(self.i).copied() }
    fn ws(&mut self) { while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') { self.i += 1; } }

    fn value(&mut self) -> Result<Json, JsonError> {
        self.ws();
        match self.peek() {
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => self.string().map(Json::Str),
            Some(b't') | Some(b'f') => self.boolean(),
            Some(b'n') => self.null(),
            Some(c) if c == b'-' || c.is_ascii_digit() => self.number(),
            _ => Err(self.err("expected value")),
        }
    }

    fn object(&mut self) -> Result<Json, JsonError> {
        self.i += 1; // {
        let mut out = Vec::new();
        self.ws();
        if self.peek() == Some(b'}') { self.i += 1; return Ok(Json::Object(out)); }
        loop {
            self.ws();
            if self.peek() != Some(b'"') { return Err(self.err("expected string key")); }
            let k = self.string()?;
            self.ws();
            if self.peek() != Some(b':') { return Err(self.err("expected ':'")); }
            self.i += 1;
            let v = self.value()?;
            out.push((k, v));
            self.ws();
            match self.peek() {
                Some(b',') => { self.i += 1; }
                Some(b'}') => { self.i += 1; break; }
                _ => return Err(self.err("expected ',' or '}'")),
            }
        }
        Ok(Json::Object(out))
    }

    fn array(&mut self) -> Result<Json, JsonError> {
        self.i += 1; // [
        let mut out = Vec::new();
        self.ws();
        if self.peek() == Some(b']') { self.i += 1; return Ok(Json::Array(out)); }
        loop {
            let v = self.value()?;
            out.push(v);
            self.ws();
            match self.peek() {
                Some(b',') => { self.i += 1; }
                Some(b']') => { self.i += 1; break; }
                _ => return Err(self.err("expected ',' or ']'")),
            }
        }
        Ok(Json::Array(out))
    }

    fn string(&mut self) -> Result<String, JsonError> {
        if self.peek() != Some(b'"') { return Err(self.err("expected '\"'")); }
        self.i += 1;
        let mut out = String::new();
        while let Some(c) = self.peek() {
            match c {
                b'"' => { self.i += 1; return Ok(out); }
                b'\\' => {
                    self.i += 1;
                    let e = self.peek().ok_or_else(|| self.err("unterminated escape"))?;
                    self.i += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{0008}'),
                        b'f' => out.push('\u{000C}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let cp = self.hex4()?;
                            if (0xD800..=0xDBFF).contains(&cp) {
                                // high surrogate; expect a low surrogate next
                                if self.peek() == Some(b'\\') {
                                    self.i += 1;
                                    if self.peek() == Some(b'u') {
                                        self.i += 1;
                                        let lo = self.hex4()?;
                                        if (0xDC00..=0xDFFF).contains(&lo) {
                                            let c = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                                            out.push(char::from_u32(c).unwrap_or('\u{FFFD}'));
                                            continue;
                                        }
                                    }
                                }
                                out.push('\u{FFFD}');
                            } else {
                                out.push(char::from_u32(cp).unwrap_or('\u{FFFD}'));
                            }
                        }
                        _ => return Err(self.err(format!("bad escape \\{}", e as char))),
                    }
                }
                _ => {
                    // Copy one UTF-8 char (find its end). The byte is c; valid UTF-8 continuation
                    // bytes follow for multibyte sequences.
                    let start = self.i;
                    let len = utf8_len(c);
                    self.i += len;
                    out.push_str(std::str::from_utf8(&self.b[start..self.i]).unwrap_or("\u{FFFD}"));
                }
            }
        }
        Err(self.err("unterminated string"))
    }

    fn hex4(&mut self) -> Result<u32, JsonError> {
        let mut v = 0u32;
        for _ in 0..4 {
            let c = self.peek().ok_or_else(|| self.err("short \\uXXXX"))?;
            self.i += 1;
            v = v * 16 + (c as char).to_digit(16).ok_or_else(|| self.err("bad hex digit"))?;
        }
        Ok(v)
    }

    fn boolean(&mut self) -> Result<Json, JsonError> {
        if self.b[self.i..].starts_with(b"true") { self.i += 4; Ok(Json::Bool(true)) }
        else if self.b[self.i..].starts_with(b"false") { self.i += 5; Ok(Json::Bool(false)) }
        else { Err(self.err("expected true/false")) }
    }

    fn null(&mut self) -> Result<Json, JsonError> {
        if self.b[self.i..].starts_with(b"null") { self.i += 4; Ok(Json::Null) }
        else { Err(self.err("expected null")) }
    }

    fn number(&mut self) -> Result<Json, JsonError> {
        let start = self.i;
        if self.peek() == Some(b'-') { self.i += 1; }
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() { self.i += 1; } else { break; }
        }
        if self.peek() == Some(b'.') {
            self.i += 1;
            while let Some(c) = self.peek() { if c.is_ascii_digit() { self.i += 1; } else { break; } }
        }
        if matches!(self.peek(), Some(b'e') | Some(b'E')) {
            self.i += 1;
            if matches!(self.peek(), Some(b'+') | Some(b'-')) { self.i += 1; }
            while let Some(c) = self.peek() { if c.is_ascii_digit() { self.i += 1; } else { break; } }
        }
        let s = std::str::from_utf8(&self.b[start..self.i]).map_err(|_| self.err("non-utf8 number"))?;
        Ok(Json::Num(s.to_string()))
    }
}

fn utf8_len(b: u8) -> usize {
    if b < 0x80 { 1 } else if b >> 5 == 0b110 { 2 } else if b >> 4 == 0b1110 { 3 } else { 4 }
}

/// Emit `Json` as `json.dumps(value, ensure_ascii=False)` would: separators `', '` / `': '`, key
/// insertion order, `/` unescaped, non-ASCII raw, control chars as `\u00XX` / named escapes.
pub fn dumps(v: &Json) -> String {
    let mut out = String::new();
    dumps_into(v, &mut out);
    out
}

fn dumps_into(v: &Json, out: &mut String) {
    match v {
        Json::Null => out.push_str("null"),
        Json::Bool(true) => out.push_str("true"),
        Json::Bool(false) => out.push_str("false"),
        Json::Num(s) => out.push_str(s),
        Json::Str(s) => { out.push('"'); escape_json_string(s, out); out.push('"'); }
        Json::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 { out.push_str(", "); }
                dumps_into(x, out);
            }
            out.push(']');
        }
        Json::Object(o) => {
            out.push('{');
            for (i, (k, x)) in o.iter().enumerate() {
                if i > 0 { out.push_str(", "); }
                out.push('"'); escape_json_string(k, out); out.push('"');
                out.push_str(": ");
                dumps_into(x, out);
            }
            out.push('}');
        }
    }
}

/// Python `json.dumps` string escaping (ensure_ascii=False): quote/backslash + the named control
/// escapes, `\u00XX` for the rest of C0; everything else (incl. `/` and all of UTF-8) raw.
fn escape_json_string(s: &str, out: &mut String) {
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{0008}' => out.push_str("\\b"),
            '\u{000C}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
}

// =================================================================================================
// Special tokens & templates — copied verbatim from encoding_dsv4.py (full-width U+FF5C/U+2581)
// =================================================================================================

pub const BOS_TOKEN: &str = "<｜begin▁of▁sentence｜>";
pub const EOS_TOKEN: &str = "<｜end▁of▁sentence｜>";
pub const THINKING_START_TOKEN: &str = "<think>";
pub const THINKING_END_TOKEN: &str = "</think>";
/// The DSML marker is BARS-ONLY (`｜DSML｜`); the templates add the angle brackets around it
/// (verified: id 128825, `special:false`). This is the one the tokenizer matches as one token.
pub const DSML_TOKEN: &str = "｜DSML｜";

pub const USER_SP_TOKEN: &str = "<｜User｜>";
pub const ASSISTANT_SP_TOKEN: &str = "<｜Assistant｜>";
pub const LATEST_REMINDER_SP_TOKEN: &str = "<｜latest_reminder｜>";

/// Quick-instruction task tokens (internal classification). `action` has special transition rules;
/// the others append directly after the user content (see render_message).
pub fn task_sp_token(task: &str) -> Option<&'static str> {
    Some(match task {
        "action" => "<｜action｜>",
        "query" => "<｜query｜>",
        "authority" => "<｜authority｜>",
        "domain" => "<｜domain｜>",
        "title" => "<｜title｜>",
        "read_url" => "<｜read_url｜>",
        _ => return None,
    })
}

/// Reasoning effort levels (mirrors `encoding_dsv4.py` REASONING_EFFORT_PROMPTS). In thinking
/// mode, the prompt for the selected level is prepended at the very beginning of the conversation.
/// `low` is the default and adds nothing. (0731 model: REASONING_EFFORT_MAX was generalized to a
/// per-level dict; "high" previously injected NOTHING, now it injects the former max text, and
/// "max" gained a longer text.)
const REASONING_EFFORT_PROMPTS: [(&str, &str); 3] = [
    ("low", ""),
    (
        "high",
        "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n\
You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n\
Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n",
    ),
    (
        "max",
        "Reasoning Effort: Beyond maximum — exhaustive, relentless, and uncompromising.\n\
You MUST reason with the utmost depth and rigor, leaving absolutely nothing to chance: exhaustively decompose the problem into its most fundamental components, trace every causal chain to its root, and resolve the underlying cause rather than any surface symptom.\n\
Do not stop reasoning until you have independently verified the solution from multiple angles and are certain that no assumption remains unchecked and no error remains undiscovered.\n\n",
    ),
];

const DEFAULT_REASONING_EFFORT: &str = "low";

/// `encoding_dsv4.py`: `reasoning_effort = reasoning_effort or DEFAULT_REASONING_EFFORT` followed
/// by an assert membership in REASONING_EFFORT_PROMPTS (None and "" are treated as "low").
fn reasoning_effort_prompt(effort: Option<&str>) -> &'static str {
    let eff = match effort {
        Some(s) if !s.is_empty() => s,
        _ => DEFAULT_REASONING_EFFORT,
    };
    for (name, text) in REASONING_EFFORT_PROMPTS {
        if name == eff {
            return text;
        }
    }
    panic!("Invalid reasoning effort: {eff}, expected one of [\"low\", \"high\", \"max\"]");
}

const RESPONSE_FORMAT_TEMPLATE: &str =
    "## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n{schema}";

const TOOL_CALL_TEMPLATE: &str =
    "<{dsml_token}invoke name=\"{name}\">\n{arguments}\n</{dsml_token}invoke>";

const TOOL_CALLS_TEMPLATE: &str =
    "<{dsml_token}{tc_block_name}>\n{tool_calls}\n</{dsml_token}{tc_block_name}>";

const TOOL_CALLS_BLOCK_NAME: &str = "tool_calls";

const TOOL_OUTPUT_TEMPLATE: &str = "<tool_result>{content}</tool_result>";

const TOOLS_TEMPLATE: &str = "## Tools\n\n\
You have access to a set of tools to help answer the user's question. You can invoke tools by writing a \"<{dsml_token}tool_calls>\" block like the following:\n\n\
<{dsml_token}tool_calls>\n\
<{dsml_token}invoke name=\"$TOOL_NAME\">\n\
<{dsml_token}parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</{dsml_token}parameter>\n\
...\n\
</{dsml_token}invoke>\n\
<{dsml_token}invoke name=\"$TOOL_NAME2\">\n\
...\n\
</{dsml_token}invoke>\n\
</{dsml_token}tool_calls>\n\n\
String parameters should be specified as is and set `string=\"true\"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\n\
If thinking_mode is enabled (triggered by {thinking_start_token}), you MUST output your complete reasoning inside {thinking_start_token}...{thinking_end_token} BEFORE any tool calls or final response.\n\n\
Otherwise, output directly after {thinking_end_token} with tool calls or final response.\n\n\
### Available Tool Schemas\n\n\
{tool_schemas}\n\n\
You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n";

/// Single-pass `{name}` substitution (Python `str.format` semantics for the simple named-placeholders
/// these templates use — no format specs, no positional). The substituted text is NOT re-scanned, so
/// JSON braces inside a tool schema cannot collide with a placeholder name.
fn fmt_named(tmpl: &str, vars: &[(&str, &str)]) -> String {
    let mut out = String::with_capacity(tmpl.len());
    let mut rest = tmpl;
    while let Some(b) = rest.find('{') {
        out.push_str(&rest[..b]);
        match rest[b..].find('}') {
            Some(e) => {
                let name = &rest[b + 1..b + e];
                match vars.iter().find(|(n, _)| *n == name) {
                    Some((_, v)) => { out.push_str(v); rest = &rest[b + e + 1..]; }
                    None => { out.push('{'); rest = &rest[b + 1..]; } // literal '{', keep scanning
                }
            }
            None => { out.push('{'); rest = &rest[b + 1..]; }
        }
    }
    out.push_str(rest);
    out
}

// =================================================================================================
// Utility: OpenAI → internal conversions, tool rendering
// =================================================================================================

/// `tools_from_openai_format`: `[{"type":"function","function":{...}}, ...]` → the function objects.
fn tools_from_openai_format(tools: &Json) -> Vec<Json> {
    tools.as_array().map(|a| a.iter().filter_map(|t| t.get("function").cloned()).collect()).unwrap_or_default()
}

/// `tool_calls_from_openai_format`: pull `name`+`arguments` (a JSON string) out of each call.
/// Returns ordered objects `{name, arguments}` preserving only the two keys the encoder reads.
fn tool_calls_from_openai_format(tool_calls: &Json) -> Vec<Json> {
    tool_calls.as_array().map(|a| {
        a.iter().filter_map(|tc| {
            let f = tc.get("function")?;
            Some(Json::Object(vec![
                ("name".into(), f.get("name").cloned().unwrap_or(Json::Null)),
                ("arguments".into(), f.get("arguments").cloned().unwrap_or(Json::Null)),
            ]))
        }).collect()
    }).unwrap_or_default()
}

/// `encode_arguments_to_dsml`: emit one `<｜DSML｜parameter …>` line per argument.
///
/// Typing rule: Python `str` → `string="true"`, value VERBATIM; everything else → `string="false"`
/// and the value is `json.dumps(ensure_ascii=False)`. If the arguments string isn't valid JSON it
/// degrades to a single param named `arguments` carrying the raw string (string="true").
fn encode_arguments_to_dsml(tool_call: &Json) -> String {
    let args_str = tool_call.get("arguments").and_then(|v| v.as_str()).unwrap_or("");
    let parsed = parse_json(args_str);
    let arguments: Vec<(String, Json)> = match parsed {
        Ok(Json::Object(o)) => o,
        _ => vec![("arguments".to_string(), Json::Str(args_str.to_string()))],
    };
    let mut lines = Vec::with_capacity(arguments.len());
    for (k, v) in arguments {
        let (is_str, value) = match &v {
            Json::Str(s) => (true, s.clone()),
            other => (false, dumps(other)),
        };
        lines.push(fmt_named(
            "<{dsml_token}parameter name=\"{key}\" string=\"{is_str}\">{value}</{dsml_token}parameter>",
            &[
                ("dsml_token", DSML_TOKEN),
                ("key", &k),
                ("is_str", if is_str { "true" } else { "false" }),
                ("value", &value),
            ],
        ));
    }
    lines.join("\n")
}

/// `render_tools`: the `## Tools` block. Schemas = the function objects, one `json.dumps` per line.
fn render_tools(tools: &[Json]) -> String {
    let schemas: Vec<String> = tools.iter().map(dumps).collect();
    fmt_named(
        TOOLS_TEMPLATE,
        &[
            ("tool_schemas", schemas.join("\n").as_str()),
            ("dsml_token", DSML_TOKEN),
            ("thinking_start_token", THINKING_START_TOKEN),
            ("thinking_end_token", THINKING_END_TOKEN),
        ],
    )
}

fn find_last_user_index(messages: &[Json]) -> isize {
    let mut last: isize = -1;
    for idx in (0..messages.len()).rev() {
        if messages[idx].get("role").and_then(|r| r.as_str()).map_or(false, |r| r == "user" || r == "developer") {
            last = idx as isize;
            break;
        }
    }
    last
}

// =================================================================================================
// Preprocessing: merge_tool_messages, sort_tool_results_by_call_order
// =================================================================================================

/// Merge standalone `role:"tool"` messages into a preceding user message's `content_blocks`
/// (DSV4 has no standalone tool role on the wire — results live as `<tool_result>` inside user).
/// User text becomes a `text` block; consecutive tool/user results coalesce into one user message.
fn merge_tool_messages(messages: &[Json]) -> Vec<Json> {
    let mut merged: Vec<Json> = Vec::with_capacity(messages.len());
    for msg in messages {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("").to_string();
        if role == "tool" {
            let tool_block = Json::Object(vec![
                ("type".into(), Json::Str("tool_result".into())),
                ("tool_use_id".into(), msg.get("tool_call_id").cloned().unwrap_or(Json::Str(String::new()))),
                ("content".into(), msg.get("content").cloned().unwrap_or(Json::Str(String::new()))),
            ]);
            push_or_merge_block(&mut merged, tool_block);
        } else if role == "user" {
            let text = msg.get("content").and_then(|c| c.as_str()).unwrap_or("").to_string();
            let text_block = Json::Object(vec![
                ("type".into(), Json::Str("text".into())),
                ("text".into(), Json::Str(text)),
            ]);
            // Only coalesce into a prior user if that prior has no `task` (mirrors .py:440-441).
            let can_merge = merged.last().map_or(false, |m| {
                m.get("role").and_then(|r| r.as_str()) == Some("user")
                    && m.get("content_blocks").is_some()
                    && m.get("task").is_none()
            });
            if can_merge {
                if let Some(Json::Object(o)) = merged.last_mut() {
                    if let Some((_, Json::Array(blocks))) = o.iter_mut().find(|(k, _)| k == "content_blocks") {
                        blocks.push(text_block);
                    }
                }
            } else {
                let mut new_msg: Vec<(String, Json)> = vec![
                    ("role".into(), Json::Str("user".into())),
                    ("content".into(), msg.get("content").cloned().unwrap_or(Json::Str(String::new()))),
                    ("content_blocks".into(), Json::Array(vec![text_block])),
                ];
                for key in ["task", "wo_eos", "mask"] {
                    if let Some(v) = msg.get(key) { new_msg.push((key.into(), v.clone())); }
                }
                merged.push(Json::Object(new_msg));
            }
        } else {
            merged.push(msg.clone());
        }
    }
    merged
}

fn push_or_merge_block(merged: &mut Vec<Json>, block: Json) {
    let can_merge = merged.last().map_or(false, |m| {
        m.get("role").and_then(|r| r.as_str()) == Some("user") && m.get("content_blocks").is_some()
    });
    if can_merge {
        if let Some(Json::Object(o)) = merged.last_mut() {
            if let Some((_, Json::Array(blocks))) = o.iter_mut().find(|(k, _)| k == "content_blocks") {
                blocks.push(block);
                return;
            }
        }
    }
    merged.push(Json::Object(vec![
        ("role".into(), Json::Str("user".into())),
        ("content_blocks".into(), Json::Array(vec![block])),
    ]));
}

/// Sort `tool_result` blocks within a user message by the order of `tool_calls` in the PRECEDING
/// assistant message (matched by id). Non-tool blocks keep their positions; ties keep stable order.
fn sort_tool_results_by_call_order(messages: &mut [Json]) {
    let mut last_order: Vec<(String, usize)> = Vec::new();
    for msg in messages.iter_mut() {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("").to_string();
        if role == "assistant" {
            if let Some(tcs) = msg.get("tool_calls").and_then(|t| t.as_array()) {
                last_order.clear();
                for (idx, tc) in tcs.iter().enumerate() {
                    let id = tc.get("id").and_then(|v| v.as_str())
                        .or_else(|| tc.get("function").and_then(|f| f.get("id")).and_then(|v| v.as_str()))
                        .unwrap_or("").to_string();
                    if !id.is_empty() { last_order.push((id, idx)); }
                }
            }
        } else if role == "user" {
            // Work on a clone of content_blocks; if it needs reordering, write it back.
            let blocks = match msg.get("content_blocks").and_then(|b| b.as_array()) {
                Some(b) => b.to_vec(),
                None => continue,
            };
            let tool_idx: Vec<usize> = blocks.iter().enumerate()
                .filter_map(|(i, b)| {
                    let is_tr = b.get("type").and_then(|t| t.as_str()) == Some("tool_result");
                    if is_tr { Some(i) } else { None }
                }).collect();
            if tool_idx.len() > 1 && !last_order.is_empty() {
                let mut keyed: Vec<(usize, Json)> = tool_idx.iter()
                    .map(|&i| {
                        let bid = blocks[i].get("tool_use_id").and_then(|v| v.as_str()).unwrap_or("").to_string();
                        let key = last_order.iter().find_map(|(id, pos)| if *id == bid { Some(*pos) } else { None }).unwrap_or(0);
                        (key, blocks[i].clone())
                    }).collect();
                keyed.sort_by_key(|(k, _)| *k);
                let mut sorted_iter = keyed.into_iter();
                let new_blocks: Vec<Json> = blocks.iter().map(|b| {
                    if b.get("type").and_then(|t| t.as_str()) == Some("tool_result") {
                        sorted_iter.next().map(|(_, v)| v).unwrap_or_else(|| b.clone())
                    } else { b.clone() }
                }).collect();
                if let Json::Object(o) = msg {
                    if let Some((_, arr)) = o.iter_mut().find(|(k, _)| k == "content_blocks") {
                        *arr = Json::Array(new_blocks);
                    }
                }
            }
        }
    }
}

/// `_drop_thinking_messages`: keep user/system/tool/latest_reminder/direct_search_results always
/// (as-is); keep everything at/after the last user index (as-is, reasoning intact); assistant turns
/// BEFORE the last user are kept but have `reasoning_content` stripped; developer/other turns before
/// the last user are dropped entirely. (Mirrors encoding_dsv4.py:575-599 exactly.)
fn drop_thinking_messages(messages: &[Json]) -> Vec<Json> {
    let last_user = find_last_user_index(messages);
    let keep_roles: &[&str] = &["user", "system", "tool", "latest_reminder", "direct_search_results"];
    let mut out = Vec::new();
    for (idx, msg) in messages.iter().enumerate() {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("").to_string();
        let keep_as_is = keep_roles.contains(&role.as_str()) || (idx as isize) >= last_user;
        if keep_as_is {
            out.push(msg.clone());
        } else if role == "assistant" {
            // strip reasoning_content only (preserve order of the remaining keys)
            if let Json::Object(o) = msg {
                let filtered: Vec<(String, Json)> = o.iter()
                    .filter(|(k, _)| k != "reasoning_content").map(|(k, v)| (k.clone(), v.clone())).collect();
                out.push(Json::Object(filtered));
            } else {
                out.push(msg.clone());
            }
        }
        // developer / unknown roles before the last user index are dropped.
    }
    out
}

// =================================================================================================
// render_message — the core (encoding_dsv4.py:223-394)
// =================================================================================================

fn render_message(
    index: usize,
    messages: &[Json],
    thinking_mode: &str,
    drop_thinking: bool,
    reasoning_effort: Option<&str>,
) -> String {
    debug_assert!(thinking_mode == "chat" || thinking_mode == "thinking");
    let mut prompt = String::new();
    let msg = &messages[index];
    let last_user_idx = find_last_user_index(messages);

    let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("").to_string();
    let content = msg.get("content").and_then(|c| c.as_str());
    let tools_raw = msg.get("tools");
    let response_format = msg.get("response_format");
    let tool_calls_raw = msg.get("tool_calls");
    let reasoning_content = msg.get("reasoning_content").and_then(|r| r.as_str());
    let wo_eos = msg.get("wo_eos").and_then(|v| v.as_bool()).unwrap_or(false);

    let tools: Vec<Json> = tools_raw.map(tools_from_openai_format).unwrap_or_default();
    let tool_calls: Vec<Json> = tool_calls_raw.map(tool_calls_from_openai_format).unwrap_or_default();

    // Reasoning effort prefix (only at index 0 in thinking mode; "low" adds nothing).
    if index == 0 && thinking_mode == "thinking" {
        prompt.push_str(reasoning_effort_prompt(reasoning_effort));
    }

    if role == "system" {
        prompt.push_str(content.unwrap_or(""));
        if !tools.is_empty() {
            prompt.push_str("\n\n");
            prompt.push_str(&render_tools(&tools));
        }
        if let Some(rf) = response_format {
            prompt.push_str("\n\n");
            prompt.push_str(&fmt_named(RESPONSE_FORMAT_TEMPLATE, &[("schema", dumps(rf).as_str())]));
        }
    } else if role == "developer" {
        // developer renders AS a user message (search-agent role); USER_SP prefix then content (+tools).
        let c = content.unwrap_or("").to_string();
        let mut cd = String::from(USER_SP_TOKEN);
        cd.push_str(&c);
        if !tools.is_empty() { cd.push_str("\n\n"); cd.push_str(&render_tools(&tools)); }
        if let Some(rf) = response_format { cd.push_str("\n\n"); cd.push_str(&fmt_named(RESPONSE_FORMAT_TEMPLATE, &[("schema", dumps(rf).as_str())])); }
        prompt.push_str(&cd);
    } else if role == "user" {
        prompt.push_str(USER_SP_TOKEN);
        if let Some(Json::Array(blocks)) = msg.get("content_blocks") {
            let mut parts: Vec<String> = Vec::with_capacity(blocks.len());
            for block in blocks {
                let bt = block.get("type").and_then(|t| t.as_str()).unwrap_or("");
                if bt == "text" {
                    parts.push(block.get("text").and_then(|t| t.as_str()).unwrap_or("").to_string());
                } else if bt == "tool_result" {
                    let mut tool_content = block.get("content").and_then(|c| c.as_str()).unwrap_or("").to_string();
                    // tool_result content may itself be a list of {type,text} blocks.
                    if let Some(Json::Array(arr)) = block.get("content") {
                        let mut text_parts: Vec<String> = Vec::new();
                        for b in arr {
                            if b.get("type").and_then(|t| t.as_str()) == Some("text") {
                                text_parts.push(b.get("text").and_then(|t| t.as_str()).unwrap_or("").to_string());
                            } else {
                                let ut = b.get("type").and_then(|t| t.as_str()).unwrap_or("");
                                text_parts.push(format!("[Unsupported {}]", ut));
                            }
                        }
                        tool_content = text_parts.join("\n\n");
                    }
                    parts.push(fmt_named(TOOL_OUTPUT_TEMPLATE, &[("content", tool_content.as_str())]));
                } else {
                    parts.push(format!("[Unsupported {}]", bt));
                }
            }
            prompt.push_str(&parts.join("\n\n"));
        } else {
            prompt.push_str(content.unwrap_or(""));
        }
    } else if role == "latest_reminder" {
        prompt.push_str(LATEST_REMINDER_SP_TOKEN);
        prompt.push_str(content.unwrap_or(""));
    } else if role == "tool" {
        // Should never reach here after merge_tool_messages; mirror the .py's NotImplementedError.
        panic!("dsv4_chat: role 'tool' must be preprocessed by merge_tool_messages");
    } else if role == "assistant" {
        let mut thinking_part = String::new();
        let mut tc_content = String::new();

        if !tool_calls.is_empty() {
            let tc_list: Vec<String> = tool_calls.iter().map(|tc| {
                let name = tc.get("name").and_then(|n| n.as_str()).unwrap_or("");
                let args = encode_arguments_to_dsml(tc);
                fmt_named(TOOL_CALL_TEMPLATE, &[
                    ("dsml_token", DSML_TOKEN), ("name", name), ("arguments", args.as_str()),
                ])
            }).collect();
            tc_content.push_str("\n\n");
            tc_content.push_str(&fmt_named(TOOL_CALLS_TEMPLATE, &[
                ("dsml_token", DSML_TOKEN),
                ("tool_calls", tc_list.join("\n").as_str()),
                ("tc_block_name", TOOL_CALLS_BLOCK_NAME),
            ]));
        }

        let summary_content = content.unwrap_or("").to_string();
        let rc = reasoning_content.unwrap_or("").to_string();

        // A task on the PREVIOUS message means this assistant turn is a task output (no thinking).
        let prev_has_task = index > 0 && messages[index - 1].get("task").is_some();

        if thinking_mode == "thinking" && !prev_has_task {
            if !drop_thinking || (index as isize) > last_user_idx {
                thinking_part.push_str(&rc);
                thinking_part.push_str(THINKING_END_TOKEN);
            }
        }

        // assistant_msg_template = "{reasoning}{content}{tool_calls}" + EOS (wo_eos drops the EOS).
        prompt.push_str(&fmt_named(
            "{reasoning}{content}{tool_calls}",
            &[("reasoning", thinking_part.as_str()), ("content", summary_content.as_str()), ("tool_calls", tc_content.as_str())],
        ));
        if !wo_eos {
            prompt.push_str(EOS_TOKEN);
        }
    } else {
        panic!("dsv4_chat: unknown role '{}'", role);
    }

    // Transition tokens based on what FOLLOWS. If the next message is not assistant/latest_reminder,
    // this message is followed by a user/system block and gets NO generation-prompt transition.
    if index + 1 < messages.len() {
        let next_role = messages[index + 1].get("role").and_then(|r| r.as_str()).unwrap_or("");
        if next_role != "assistant" && next_role != "latest_reminder" {
            return prompt;
        }
    }

    // A task on THIS message wins over the normal generation-prompt transition.
    if let Some(task) = msg.get("task").and_then(|t| t.as_str()) {
        let sp = task_sp_token(task).unwrap_or_else(|| panic!("dsv4_chat: invalid task '{}'", task));
        if task != "action" {
            prompt.push_str(sp);
        } else {
            // action: Assistant + thinking token (end in chat, start in thinking) + action sp token.
            prompt.push_str(ASSISTANT_SP_TOKEN);
            prompt.push_str(if thinking_mode != "thinking" { THINKING_END_TOKEN } else { THINKING_START_TOKEN });
            prompt.push_str(sp);
        }
    } else if role == "user" || role == "developer" {
        prompt.push_str(ASSISTANT_SP_TOKEN);
        if !drop_thinking && thinking_mode == "thinking" {
            prompt.push_str(THINKING_START_TOKEN);
        } else if drop_thinking && thinking_mode == "thinking" && (index as isize) >= last_user_idx {
            prompt.push_str(THINKING_START_TOKEN);
        } else {
            prompt.push_str(THINKING_END_TOKEN);
        }
    }

    prompt
}

// =================================================================================================
// encode_messages — the main entry point (encoding_dsv4.py:506-572)
// =================================================================================================

#[derive(Debug, Clone)]
pub struct EncodeOptions {
    pub thinking_mode: String,
    pub context: Vec<Json>,
    pub drop_thinking: bool,
    pub add_default_bos_token: bool,
    pub reasoning_effort: Option<String>,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self {
            thinking_mode: "thinking".into(),
            context: vec![],
            drop_thinking: true,
            add_default_bos_token: true,
            reasoning_effort: None,
        }
    }
}

/// Encode a conversation into the DSV4 wire string (byte-exact vs `encoding_dsv4.encode_messages`).
pub fn encode_messages(messages: &[Json], opts: &EncodeOptions) -> String {
    let thinking_mode = opts.thinking_mode.as_str();

    // Preprocess, mirroring the .py exactly (the context branch is not exercised by the golden
    // vectors — they pass context=None — but is kept faithful for the multi-chunk serving path):
    //   messages = merge_tool_messages(messages)
    //   messages = sort_tool_results_by_call_order(context + messages)[len(context):]
    //   if context: context = merge_tool_messages(sort_tool_results_by_call_order(context))
    let mut messages_merged = merge_tool_messages(messages);
    let mut context = opts.context.clone(); // un-merged original, for the combined sort + the slice
    {
        let mut combined = context.clone();
        combined.extend_from_slice(&messages_merged);
        sort_tool_results_by_call_order(&mut combined);
        messages_merged = combined[context.len()..].to_vec();
    }
    if !context.is_empty() {
        context = merge_tool_messages(&context);
        sort_tool_results_by_call_order(&mut context);
    }

    let mut full_messages = context.clone();
    full_messages.extend_from_slice(&messages_merged);

    // BOS from the encoding layer (add_bos_token:false at the tokenizer level), first chunk only.
    let mut prompt = if opts.add_default_bos_token && context.is_empty() { String::from(BOS_TOKEN) } else { String::new() };

    // Tools anywhere → drop_thinking is forced off (interleaved reasoning must survive across calls).
    let mut effective_drop_thinking = opts.drop_thinking;
    if full_messages.iter().any(|m| m.get("tools").is_some()) {
        effective_drop_thinking = false;
    }

    let (num_to_render, context_len) = if thinking_mode == "thinking" && effective_drop_thinking {
        let dropped_full = drop_thinking_messages(&full_messages);
        let dropped_ctx = drop_thinking_messages(&context);
        let num_to_render = dropped_full.len().saturating_sub(dropped_ctx.len());
        let context_len = dropped_full.len().saturating_sub(num_to_render);
        full_messages = dropped_full;
        (num_to_render, context_len)
    } else {
        (messages_merged.len(), context.len())
    };

    for idx in 0..num_to_render {
        prompt.push_str(&render_message(
            idx + context_len,
            &full_messages,
            thinking_mode,
            effective_drop_thinking,
            opts.reasoning_effort.as_deref(),
        ));
    }
    prompt
}

// =================================================================================================
// Output parsing (encoding_dsv4.py:602-744) — strict spec version + a lenient server variant
// =================================================================================================

/// Read from `index` until the earliest of `stops`; return (new_index, content_before, matched_stop).
fn read_until_stop(text: &str, index: usize, stops: &[&str]) -> (usize, String, Option<&'static str>) {
    let bytes = text.as_bytes();
    let from = index.min(bytes.len());
    let mut min_pos = bytes.len();
    let mut matched: Option<&'static str> = None;
    for s in stops {
        if let Some(rel) = text[from..].find(s) {
            let pos = from + rel;
            if pos < min_pos { min_pos = pos; matched = Some(s_static(s)); }
        }
    }
    match matched {
        Some(s) => {
            let content = text[from..min_pos].to_string();
            (min_pos + s.len(), content, Some(s))
        }
        None => (bytes.len(), text[from..].to_string(), None),
    }
}

/// Best-effort static-lifetime mirror of a literal (all our stops are &'static).
fn s_static(s: &str) -> &'static str {
    for cand in [
        THINKING_END_TOKEN, EOS_TOKEN, DSML_TOKEN, "<｜DSML｜tool_calls", "</｜DSML｜tool_calls>",
        "<｜DSML｜invoke", "</｜DSML｜invoke", "<｜DSML｜parameter", "/｜DSML｜parameter",
    ] {
        if cand == s { return cand; }
    }
    // Fallbacks used only by callers with dynamic strings — leak a stable copy (parsing is rare).
    leaked(s)
}
fn leaked(s: &str) -> &'static str { Box::leak(s.to_string().into_boxed_str()) }

/// Result of parsing a model completion into a structured assistant turn.
#[derive(Debug, Clone, Default)]
pub struct ParsedAssistant {
    pub content: String,
    pub reasoning_content: String,
    /// OpenAI-format tool calls: `{type:"function", function:{name, arguments(JSON string)}}`.
    pub tool_calls: Vec<Json>,
}

/// Decode DSML parameters back to an arguments JSON string (mirrors decode_dsml_to_arguments).
/// `tool_args`: param_name -> (value, is_string_flag), in encounter order.
fn decode_dsml_to_arguments(tool_name: &str, tool_args: &[(String, String, bool)]) -> Json {
    // Each entry: f"{to_json(key)}: {value}" where value is to_json(value) if string=true else raw.
    let parts: Vec<String> = tool_args.iter().map(|(k, v, is_str)| {
        let key_json = dumps(&Json::Str(k.clone()));
        let val = if *is_str { dumps(&Json::Str(v.clone())) } else { v.clone() };
        format!("{}: {}", key_json, val)
    }).collect();
    let arguments = format!("{{ {} }}", parts.join(", "));
    Json::Object(vec![
        ("name".into(), Json::Str(tool_name.into())),
        ("arguments".into(), Json::Str(arguments)),
    ])
}

/// Parse DSML tool calls out of model text (strict — mirrors parse_tool_calls:630-684). On any
/// structural error returns what was collected so far (the lenient server path degrades gracefully).
fn parse_tool_calls_strict(index: usize, text: &str) -> (usize, Option<&'static str>, Vec<Json>) {
    let mut tool_calls: Vec<Json> = Vec::new();
    let mut stop_token: Option<&'static str> = None;
    let tool_calls_end = "</｜DSML｜tool_calls>";
    let mut idx = index;
    while idx < text.len() {
        let (new_idx, sep, st) = read_until_stop(text, idx, &["<｜DSML｜invoke", tool_calls_end]);
        idx = new_idx;
        // The .py asserts the consumed-between marker is ">\n"; be lenient if it isn't.
        if st == Some(tool_calls_end) { stop_token = st; break; }
        if st.is_none() { break; }
        stop_token = st;
        let (new_idx, name_content, st2) = read_until_stop(text, idx, &["<｜DSML｜parameter", "</｜DSML｜invoke"]);
        idx = new_idx;
        let tool_name = parse_name_attr(&name_content).unwrap_or_default();
        let mut tool_args: Vec<(String, String, bool)> = Vec::new();
        let mut cur_stop = st2;
        while cur_stop == Some("<｜DSML｜parameter") {
            let (new_idx, param_content, pst) = read_until_stop(text, idx, &["/｜DSML｜parameter"]);
            idx = new_idx;
            if let Some((pname, pstring, pvalue)) = parse_param(&param_content) {
                if tool_args.iter().any(|(n, _, _)| n == &pname) {
                    // duplicate param — .py raises; we stop collecting this call.
                    break;
                }
                tool_args.push((pname, pvalue, pstring));
            }
            let (new_idx, content, nstop) = read_until_stop(text, idx, &["<｜DSML｜parameter", "</｜DSML｜invoke"]);
            idx = new_idx;
            // .py asserts content == ">\n"; lenient: ignore.
            cur_stop = nstop;
            // If the next stop is </invoke>, fall through; loop exits when cur_stop != parameter.
            if cur_stop != Some("<｜DSML｜parameter") { stop_token = cur_stop; }
        }
        let internal = decode_dsml_to_arguments(&tool_name, &tool_args);
        tool_calls.push(to_openai_tool_call(&internal));
    }
    (idx, stop_token, tool_calls)
}

fn to_openai_tool_call(internal: &Json) -> Json {
    let name = internal.get("name").and_then(|n| n.as_str()).unwrap_or("").to_string();
    let args = internal.get("arguments").and_then(|a| a.as_str()).unwrap_or("").to_string();
    Json::Object(vec![
        ("type".into(), Json::Str("function".into())),
        ("function".into(), Json::Object(vec![
            ("name".into(), Json::Str(name)),
            ("arguments".into(), Json::Str(args)),
        ])),
    ])
}

/// Extract `name="..."` from the invoke header content (the .py regex `^\s*name="(.*?)">\n$`).
fn parse_name_attr(content: &str) -> Option<String> {
    let i = content.find("name=\"")? + "name=\"".len();
    let rest = &content[i..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

/// Extract (name, string_flag, value) from a parameter's between-markers content.
/// .py regex: `^ name="(.*?)" string="(true|false)">(.*?)<$` (DOTALL).
fn parse_param(content: &str) -> Option<(String, bool, String)> {
    let i = content.find(" name=\"")? + " name=\"".len();
    let rest = &content[i..];
    let end = rest.find('"')?;
    let name = rest[..end].to_string();
    let rest = &rest[end + 1..];
    let si = rest.find("string=\"")? + "string=\"".len();
    let srest = &rest[si..];
    let send = srest.find('"')?;
    let is_str = srest[..send] == *"true";
    let rest = &srest[send + 1..];
    let vi = rest.find('>')? + 1;
    let vrest = &rest[vi..];
    // value runs up to the trailing '<' (the opening of </｜DSML｜parameter>).
    let vend = vrest.rfind('<')?;
    let value = vrest[..vend].to_string();
    Some((name, is_str, value))
}

/// Parse a model completion (the raw assistant turn, INCLUDING its EOS) into a structured message.
/// Mirrors `parse_message_from_completion_text`; unlike the .py this is lenient (best-effort on
/// malformed output) since the server must never crash on a stray generation.
pub fn parse_completion(text: &str, thinking_mode: &str) -> ParsedAssistant {
    let mut out = ParsedAssistant::default();
    let is_thinking = thinking_mode == "thinking";
    let tool_calls_start = "\n\n<｜DSML｜tool_calls";
    let mut index: usize = 0;

    if is_thinking {
        let (new_idx, content_delta, stop) = read_until_stop(text, index, &[THINKING_END_TOKEN, tool_calls_start]);
        index = new_idx;
        out.reasoning_content = content_delta;
        if stop != Some(THINKING_END_TOKEN) {
            // Malformed (no </think>) — return what we have; content stays empty.
            return out;
        }
    }

    let (new_idx, content_delta, stop) = read_until_stop(text, index, &[EOS_TOKEN, tool_calls_start]);
    index = new_idx;
    out.content = content_delta;
    let mut is_tool_calling = false;
    if stop == Some(tool_calls_start) {
        is_tool_calling = true;
    }

    if is_tool_calling {
        let (new_idx, _stop, tool_calls) = parse_tool_calls_strict(index, text);
        index = new_idx;
        out.tool_calls = tool_calls;
        let (_new_idx, _rest, _stop) = read_until_stop(text, index, &[EOS_TOKEN]);
    }

    // No special-token leakage into visible text (the .py asserts this; we just strip if present).
    for sp in [BOS_TOKEN, EOS_TOKEN, THINKING_START_TOKEN, THINKING_END_TOKEN, DSML_TOKEN] {
        if !sp.is_empty() {
            out.content = out.content.replace(sp, "");
            out.reasoning_content = out.reasoning_content.replace(sp, "");
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Ordered JSON: dumps must match json.dumps(ensure_ascii=False) byte-for-byte. ---
    #[test]
    fn json_dumps_matches_python() {
        let cases: &[(&str, &str)] = &[
            ("true", "true"),
            ("false", "false"),
            ("null", "null"),
            ("-1", "-1"),
            ("1.5", "1.5"),
            ("\"a/b\"", "\"a/b\""),                      // '/' NOT escaped
            ("\"a\\\"b\"", "\"a\\\"b\""),                 // '"' escaped
            ("\"café\"", "\"café\""),                     // non-ASCII raw
            ("[\"x\", 1]", "[\"x\", 1]"),                 // separators ', ' / ': '
            ("{\"b\":1,\"a\":2}", "{\"b\": 1, \"a\": 2}"), // key ORDER preserved
            ("{}", "{}"),
            ("[]", "[]"),
            ("\"\\u00e9\"", "\"é\""),                     // \uXXXX → raw UTF-8
        ];
        for (input, want) in cases {
            let v = parse_json(input).expect(input);
            assert_eq!(dumps(&v), *want, "dumps({:?}) mismatch", input);
        }
    }

    #[test]
    fn json_preserves_key_order_nested() {
        // The shape that breaks serde_json-without-preserve_order: nested objects, mixed key order.
        let input = r#"{"name":"f","parameters":{"type":"object","properties":{"z":1,"a":2},"required":["z"]}}"#;
        let v = parse_json(input).expect("parse");
        let out = dumps(&v);
        assert!(out.contains("\"z\": 1, \"a\": 2"), "nested order lost: {}", out);
        assert!(out.contains("\"name\":\"f\"") || out.contains("\"name\": \"f\""), "name missing: {}", out);
        assert_eq!(out, r#"{"name": "f", "parameters": {"type": "object", "properties": {"z": 1, "a": 2}, "required": ["z"]}}"#);
    }

    // --- fmt_named: single-pass, no re-scan of substituted text. ---
    #[test]
    fn fmt_named_basic() {
        assert_eq!(fmt_named("{a}{b}", &[("a", "1"), ("b", "2")]), "12");
        // A value containing '{a}' must NOT be re-substituted (single-pass).
        assert_eq!(fmt_named("{a}", &[("a", "{a}")]), "{a}");
        // Unknown placeholder survives as a literal '{'.
        assert_eq!(fmt_named("x{nope}y", &[("a", "1")]), "x{nope}y");
    }

    // --- encode_arguments_to_dsml typing rule (str → string="true" verbatim; else string="false"). ---
    #[test]
    fn dsml_args_typing() {
        // strings → string="true", verbatim
        let tc = Json::Object(vec![
            ("name".into(), Json::Str("get_weather".into())),
            ("arguments".into(), Json::Str(r#"{"location":"Beijing","unit":"celsius"}"#.into())),
        ]);
        let out = encode_arguments_to_dsml(&tc);
        assert_eq!(out,
            "<｜DSML｜parameter name=\"location\" string=\"true\">Beijing</｜DSML｜parameter>\n\
             <｜DSML｜parameter name=\"unit\" string=\"true\">celsius</｜DSML｜parameter>");
        // non-string → string="false", value json.dumps'd
        let tc2 = Json::Object(vec![
            ("name".into(), Json::Str("f".into())),
            ("arguments".into(), Json::Str(r#"{"count":5,"on":true,"arr":[1,2]}"#.into())),
        ]);
        let out2 = encode_arguments_to_dsml(&tc2);
        assert!(out2.contains("string=\"false\">5</"), "{}", out2);
        assert!(out2.contains("string=\"false\">true</"), "{}", out2);
        assert!(out2.contains("string=\"false\">[1, 2]</"), "{}", out2); // json.dumps separators
        // non-JSON arguments → degrade to single 'arguments' param (string="true")
        let tc3 = Json::Object(vec![
            ("name".into(), Json::Str("f".into())),
            ("arguments".into(), Json::Str("not json".into())),
        ]);
        let out3 = encode_arguments_to_dsml(&tc3);
        assert!(out3.contains("name=\"arguments\" string=\"true\">not json</"), "{}", out3);
    }

    // --- Output parser: round-trip a synthesized thinking + content completion. ---
    #[test]
    fn parse_thinking_content() {
        let text = format!("Reasoning here.{}Final answer.{}", THINKING_END_TOKEN, EOS_TOKEN);
        let p = parse_completion(&text, "thinking");
        assert_eq!(p.reasoning_content, "Reasoning here.");
        assert_eq!(p.content, "Final answer.");
        assert!(p.tool_calls.is_empty());
    }

    #[test]
    fn parse_tool_call_round_trip() {
        // Synthesize a strict-grammar DSML tool call + EOS, then parse it back.
        let text = format!(
            "why not.{}\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_weather\">\n<｜DSML｜parameter name=\"location\" string=\"true\">Beijing</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>{}",
            THINKING_END_TOKEN, EOS_TOKEN
        );
        let p = parse_completion(&text, "thinking");
        assert_eq!(p.reasoning_content, "why not.");
        assert_eq!(p.content, "");
        assert_eq!(p.tool_calls.len(), 1, "{:?}", p.tool_calls);
        let tc = &p.tool_calls[0];
        assert_eq!(tc.get("function").unwrap().get("name").unwrap().as_str(), Some("get_weather"));
        let args = tc.get("function").unwrap().get("arguments").unwrap().as_str().unwrap();
        // arguments round-trips to a JSON object with location=Beijing
        let reparsed = parse_json(args).expect("arguments must be JSON");
        assert_eq!(reparsed.get("location").unwrap().as_str(), Some("Beijing"));
    }

    // --- Reasoning effort (0731 semantics: REASONING_EFFORT_PROMPTS{low,high,max}, prepended ---
    // --- at index 0 in thinking mode; "low" is the default and adds nothing)             ---
    fn simple_msgs(content: &str) -> Vec<Json> {
        vec![Json::Object(vec![
            ("role".into(), Json::Str("user".into())),
            ("content".into(), Json::Str(content.into())),
        ])]
    }

    #[test]
    fn reasoning_effort_prefix_semantics() {
        let msgs = simple_msgs("hello");
        let plain = render_message(0, &msgs, "thinking", false, None);
        // low / None / "" add nothing (byte-identical to the pre-0731 default path; "" is
        // falsy in the .py's `reasoning_effort or DEFAULT_REASONING_EFFORT`).
        assert_eq!(render_message(0, &msgs, "thinking", false, Some("low")), plain);
        assert_eq!(render_message(0, &msgs, "thinking", false, None), plain);
        assert_eq!(render_message(0, &msgs, "thinking", false, Some("")), plain);

        // high prepends the former REASONING_EFFORT_MAX text (0731: 'high' no longer injects nothing).
        let high_text = REASONING_EFFORT_PROMPTS[1].1;
        let high = render_message(0, &msgs, "thinking", false, Some("high"));
        assert!(high.starts_with(high_text), "high prefix missing");
        assert_eq!(&high[high_text.len()..], plain, "high must be prefix + unchanged message");

        // max prepends the new longer text, distinct from high (0731 semantics).
        let max_text = REASONING_EFFORT_PROMPTS[2].1;
        let mx = render_message(0, &msgs, "thinking", false, Some("max"));
        assert!(mx.starts_with(max_text), "max prefix missing");
        assert_eq!(&mx[max_text.len()..], plain, "max must be prefix + unchanged message");
        assert!(max_text != high_text, "max and high must differ (0731 semantics)");

        // chat mode: no prefix at any level (compare against the chat-mode baseline — the
        // transition token differs from thinking mode).
        let plain_chat = render_message(0, &msgs, "chat", false, None);
        assert_eq!(render_message(0, &msgs, "chat", false, Some("max")), plain_chat);
        assert_eq!(render_message(0, &msgs, "chat", false, Some("high")), plain_chat);

        // non-zero index: no prefix.
        let two = vec![
            simple_msgs("first")[0].clone(),
            simple_msgs("second")[0].clone(),
        ];
        assert_eq!(render_message(1, &two, "thinking", false, Some("max")),
                   render_message(1, &two, "thinking", false, None));
    }

    #[test]
    #[should_panic(expected = "Invalid reasoning effort")]
    fn reasoning_effort_invalid_panics() {
        // .py asserts membership in REASONING_EFFORT_PROMPTS; the port must error, not silently
        // inject nothing.
        let msgs = simple_msgs("hello");
        render_message(0, &msgs, "thinking", false, Some("medium"));
    }

    // The golden-vector gate lives in tests/dsv4_chat_test.rs (it reads the bundle's test_input/
    // output files and asserts byte counts 2390/342/3313/2552 + sha256). This module covers the pieces.
}
