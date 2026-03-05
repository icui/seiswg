#!/bin/bash
cargo build --release
cd web
wasm-pack build --release --target web --out-dir web/pkg