# HOTFIX `serve-content-parts` — committed request fixtures

Committed fixtures for the OpenAI multipart `content` hotfix (branch `serve-content-parts`).
Each JSON payload is a full `POST /v1/chat/completions` body. The authoring bug: the request
schema deserialized `messages[N].content` as `Option<String>`, so the ARRAY form agent
clients send (`[{"type":"text",...}]`) failed with a confusing
`422 invalid type: sequence, expected a string`.

| fixture | shape | expected |
|---|---|---|
| `a_plain_string.json` | plain string content | 200 (regression: unchanged) |
| `b_single_text_part.json` | single-text-part array (**the Pi payload shape**) | 200 |
| `c_multi_text_part.json` | two text parts | 200 (joined with `\n`) |
| `d_null_assistant_content.json` | assistant `content: null` + `tool_calls` | 200 |
| `e_image_url_part.json` | array with an `image_url` part | **422 + `image content parts are not supported by this build`** |
| `f_agent_conversation.json` | system + multi-turn + tool messages | 200 |

`run_fixtures.sh` POSTs all six to a served endpoint and asserts the status + body marker
(asserting the SUCCESS signal, not merely the absence of errors):

```bash
BASE_URL=http://127.0.0.1:9000/v1 ./run_fixtures.sh
```

The `image_url` rejection returns a CLEAR, actionable 422 (not a serde type error) and is
the designated vision entry point: when the image tower lands, that branch becomes the
dispatch into image preprocessing instead of rejecting the request.

Headless coverage of the same shapes lives in the Rust unit test
`tokenizer::tests::content_accepts_string_array_and_null` (`src/tokenizer.rs`).
