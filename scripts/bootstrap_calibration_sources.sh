#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
    echo "usage: $0 SOURCE_ROOT" >&2
    exit 2
fi

source_root=$1
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
mkdir -p "$source_root"

fetch_file() {
    url=$1
    destination=$2
    expected_sha256=$3
    if [ ! -f "$destination" ]; then
        command -v curl >/dev/null 2>&1 || {
            echo "curl is required to download $url" >&2
            exit 1
        }
        mkdir -p "$(dirname -- "$destination")"
        partial="$destination.part"
        if [ -e "$partial" ]; then
            echo "refusing to overwrite incomplete download: $partial" >&2
            exit 1
        fi
        curl --fail --location --retry 3 --output "$partial" "$url"
        actual=$(sha256sum "$partial" | cut -d ' ' -f 1)
        if [ "$actual" != "$expected_sha256" ]; then
            echo "checksum mismatch for $url" >&2
            echo "expected $expected_sha256, got $actual" >&2
            exit 1
        fi
        mv "$partial" "$destination"
    fi
    actual=$(sha256sum "$destination" | cut -d ' ' -f 1)
    if [ "$actual" != "$expected_sha256" ]; then
        echo "checksum mismatch for existing file: $destination" >&2
        echo "expected $expected_sha256, got $actual" >&2
        exit 1
    fi
}

fetch_git() {
    url=$1
    destination=$2
    commit=$3
    if [ ! -d "$destination/.git" ]; then
        if [ -e "$destination" ]; then
            echo "refusing to replace non-git path: $destination" >&2
            exit 1
        fi
        mkdir -p "$destination"
        git -C "$destination" init --quiet
        git -C "$destination" remote add origin "$url"
        git -C "$destination" fetch --depth 1 origin "$commit"
        git -C "$destination" checkout --quiet --detach FETCH_HEAD
    fi
    actual=$(git -C "$destination" rev-parse HEAD)
    if [ "$actual" != "$commit" ]; then
        echo "git revision mismatch for $destination" >&2
        echo "expected $commit, got $actual" >&2
        exit 1
    fi
}

verify_file() {
    destination=$1
    expected_sha256=$2
    if [ ! -f "$destination" ]; then
        echo "missing generated source: $destination" >&2
        exit 1
    fi
    actual=$(sha256sum "$destination" | cut -d ' ' -f 1)
    if [ "$actual" != "$expected_sha256" ]; then
        echo "checksum mismatch for generated source: $destination" >&2
        echo "expected $expected_sha256, got $actual" >&2
        exit 1
    fi
}

fetch_file \
    "https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-train.00000-of-01024.json.gz?download=true" \
    "$source_root/c4/en/c4-train.00000-of-01024.json.gz" \
    "8ef8d75b0e045dec4aa5123a671b4564466b0707086a7ed1ba8721626dfffbc9"

fetch_file \
    "https://huggingface.co/datasets/FreedomIntelligence/alpaca-gpt4-french/resolve/main/alpaca-gpt4-french.json?download=true" \
    "$source_root/alpaca-fr/alpaca-gpt4-french.json" \
    "a4ff954113efc92131129a65153650de37a4bea217af815d61ece0d2e9b00dcc"

for language in en ja ko; do
    case "$language" in
        en) checksum=d92b92c51e8f1962a21193abe74e6f727c2bc8286035f4041505ff38a7c3ae51 ;;
        ja) checksum=2c628d3d30438c41c94443b69c38e17fd8f70d5b43b9ed7e3fb9980f56515804 ;;
        ko) checksum=2b68cfdd33a67d93c740d8ee4feaafae3f762dbf2973da04d78e61bb2cbaaab9 ;;
    esac
    fetch_file \
        "https://huggingface.co/datasets/sieu-n/alpaca_eval_multilingual/resolve/bfdcfdc68d49732e854cf531103fb3536a3317c0/alpaca_eval/$language.json?download=true" \
        "$source_root/alpaca-multilingual/alpaca_eval/$language.json" \
        "$checksum"
done

mkdir -p "$source_root/code"
fetch_git "https://github.com/golang/example.git" \
    "$source_root/code/go-example" "7f05d217867b2af52b0a28c6d1c91df97e1b5b39"
fetch_git "https://github.com/koalaman/shellcheck.git" \
    "$source_root/code/shellcheck" "9af7ee28ce587baadd950b85dd6826a16b9c068d"
fetch_git "https://github.com/microsoft/TypeScript-Website.git" \
    "$source_root/code/typescript-website" "d16dc2dc9bb11406f608c5ac1476c32a5bc806d9"

python3 "$script_dir/fetch_calibration_api.py" --output-root "$source_root"
verify_file "$source_root/aya/aya-six-languages.jsonl" \
    "f4b36c268ddd3b46fab0936511614d868f04679db07508361c640fe24840249f"
verify_file "$source_root/openr1/openr1-math-reasoning.jsonl" \
    "adbcde6cdeb5ab80f8a5a8bd5c8cb852c14d939efeca4e34328a6acc411e7227"

echo "[bootstrap] source verification complete: $source_root"
