#!/usr/bin/env bash
# HOTFIX `serve-content-parts` — POST each committed request fixture to a served OpenAI
# endpoint and assert the success signal per fixture. Usage:
#
#   BASE_URL=http://127.0.0.1:9000/v1 ./run_fixtures.sh
#
# (a) plain string        -> 200 + text content   (regression)
# (b) single text part    -> 200 + text content   (Pi payload shape)
# (c) multi text part     -> 200 + text content   (join "\n")
# (d) null assistant      -> 200 + text content   (tool_calls turn carries content:null)
# (e) image_url part      -> 422 + "image content parts are not supported by this build"
# (f) agent conversation  -> 200 + text content   (system + multi-turn + tool)
#
# Assertions are on the SUCCESS signal (status + body marker), not on the absence of errors.
set -uo pipefail
BASE_URL="${BASE_URL:-http://127.0.0.1:9000/v1}"
HERE="$(cd "$(dirname "$0")" && pwd)"
FAIL=0

run() {
  local name="$1"; local file="$2"; local expect_code="$3"; local expect_marker="$4"
  local body code
  body="$(curl -s -w '\n---HTTP:%{http_code}---' -X POST "$BASE_URL/chat/completions" \
        -H 'Content-Type: application/json' --data-binary @"$file")"
  code="$(printf '%s' "$body" | sed -n 's/.*---HTTP:\([0-9]*\)---/\1/p')"
  body="$(printf '%s' "$body" | sed 's/---HTTP:[0-9]*---$//')"
  local ok=1
  if [ "$code" != "$expect_code" ]; then ok=0; fi
  if ! printf '%s' "$body" | grep -qF "$expect_marker"; then ok=0; fi
  if [ "$ok" = "1" ]; then
    echo "PASS  $name  HTTP $code  (marker present)"
  else
    echo "FAIL  $name  HTTP $code (want $expect_code); marker '$expect_marker' missing"
    echo "      body: $(printf '%s' "$body" | head -c 300)"
    FAIL=1
  fi
}

run "a_plain_string"        "$HERE/a_plain_string.json"            200 '"content"'
run "b_single_text_part"    "$HERE/b_single_text_part.json"        200 '"content"'
run "c_multi_text_part"     "$HERE/c_multi_text_part.json"         200 '"content"'
run "d_null_assistant"      "$HERE/d_null_assistant_content.json"  200 '"content"'
run "e_image_url_part"      "$HERE/e_image_url_part.json"          422 'image content parts are not supported by this build'
run "f_agent_conversation"  "$HERE/f_agent_conversation.json"      200 '"content"'

exit $FAIL
