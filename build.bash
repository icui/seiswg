#!/bin/bash
cargo build --release
cargo test
cd web
wasm-pack build --release --target web --out-dir web/pkg