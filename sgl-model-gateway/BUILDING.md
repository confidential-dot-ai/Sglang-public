# Building the gateway (sgl-model-gateway)

The `release` profile is `opt-level="z"` + `lto="fat"` + `codegen-units=1`. That is a
size-optimised profile: the final LTO link is single-threaded and takes ~20-30 min on a
96-core box (measured 2026-09-16: 11m47s for the pyo3 wheel after an already-warm
dependency build; a cold build is worse). Nothing about the gateway needs it -- the
service is I/O-bound, and 36 MB vs 40 MB of .so is irrelevant here.

For iteration and for shipping a router fix, use the existing `ci` profile
(`lto="thin"`, `codegen-units=16`, `opt-level=2`): same runtime behaviour for this
workload, minutes instead of half an hour.

    # dev loop (fastest, unoptimised deps are already cached)
    cargo check -p sgl-model-gateway
    cargo test --lib

    # shippable wheel, fast
    cd bindings/python && maturin build --profile ci

    # size-optimised wheel (only if you actually need it)
    cd bindings/python && maturin build --release

Staging a built wheel for the deployed router (the venv at /data/sglang-venv is
read-only by design):

    rm -rf ~/router-override-images && mkdir -p ~/router-override-images
    cd ~/router-override-images && unzip -q <path-to>/sglang_router-*.whl
    # then PYTHONPATH=~/router-override-images in start-router.sh / M3_ROUTER_OVERRIDE

Note: `cargo test` (integration tests under tests/) does not compile at HEAD --
tests/common/test_app.rs and tests/wasm_test.rs predate AppState.api_keys and
AdminAuthConfig.{key_store,ops_api_key}. `cargo test --lib` is clean (459 tests).
