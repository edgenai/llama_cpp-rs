# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## v0.1.3 (2023-11-08)

### New Features

 - <csr-id-1019402eeaa6bff176a228b477486105d16d36ef/> more `async` function variants
 - <csr-id-c190df6ebfd02ef5f3e0fd50d82a456ef426e6e6/> add `LlamaSession.model`

### Other

 - <csr-id-0a0d5f3fce1c46f914b5f48802241f200538c4f7/> typo

### Commit Statistics

<csr-read-only-do-not-edit/>

 - 5 commits contributed to the release.
 - 3 commits were understood as [conventional](https://www.conventionalcommits.org).
 - 0 issues like '(#ID)' were seen in commit messages

### Commit Details

<csr-read-only-do-not-edit/>

<details><summary>view details</summary>

 * **Uncategorized**
    - Typo ([`0a0d5f3`](https://github.com/binedge/llama_cpp-rs/commit/0a0d5f3fce1c46f914b5f48802241f200538c4f7))
    - Release llama_cpp v0.1.2 ([`4d0b130`](https://github.com/binedge/llama_cpp-rs/commit/4d0b130be8f250e599908bab042431db8aa2f553))
    - More `async` function variants ([`1019402`](https://github.com/binedge/llama_cpp-rs/commit/1019402eeaa6bff176a228b477486105d16d36ef))
    - Add `LlamaSession.model` ([`c190df6`](https://github.com/binedge/llama_cpp-rs/commit/c190df6ebfd02ef5f3e0fd50d82a456ef426e6e6))
    - Release llama_cpp_sys v0.2.1, llama_cpp v0.1.1 ([`a9e5813`](https://github.com/binedge/llama_cpp-rs/commit/a9e58133cb1c1d4d45f99a7746e0af7da1a099e1))
</details>

## v0.1.2 (2023-11-08)

### New Features

 - <csr-id-dcfccdf721eb47a364cce5b1c7a54bcf94335ac0/> more `async` function variants
 - <csr-id-56285a119633682951f8748e85c6b8988e514232/> add `LlamaSession.model`

### Commit Statistics

<csr-read-only-do-not-edit/>

 - 3 commits contributed to the release.
 - 2 commits were understood as [conventional](https://www.conventionalcommits.org).
 - 0 issues like '(#ID)' were seen in commit messages

### Commit Details

<csr-read-only-do-not-edit/>

<details><summary>view details</summary>

 * **Uncategorized**
    - Release llama_cpp v0.1.2 ([`368a5de`](https://github.com/binedge/llama_cpp-rs/commit/368a5dec4379ccdbe7b68c40535f30e13f23d8c2))
    - More `async` function variants ([`dcfccdf`](https://github.com/binedge/llama_cpp-rs/commit/dcfccdf721eb47a364cce5b1c7a54bcf94335ac0))
    - Add `LlamaSession.model` ([`56285a1`](https://github.com/binedge/llama_cpp-rs/commit/56285a119633682951f8748e85c6b8988e514232))
</details>

## v0.1.1 (2023-11-08)

<csr-id-3eddbab3cc35a59acbe66fa4f5333a9ca0edb326/>

### Chore

 - <csr-id-3eddbab3cc35a59acbe66fa4f5333a9ca0edb326/> Remove debug binary from Cargo.toml

### Chore

 - <csr-id-dbdd9a4a2d813d990e5829a09fc5c8df75d9d54b/> Remove debug binary from Cargo.toml

### New Features

 - <csr-id-3bada658c9139af1c3dcdb32c60c222efb87a9f6/> add `LlamaModel::load_from_file_async`
 - <csr-id-bbf9f69a2dd068a3a20199ffce44d3c8a25b64d5/> add `LlamaModel::load_from_file_async`

### Bug Fixes

 - <csr-id-b676baa3c1a6863c7afd7a88b6f7e8ddd2a1b9bd/> require `llama_context` is accessed from behind a mutex
   This solves a race condition when several `get_completions` threads are spawned at the same time
 - <csr-id-4eb0bc9800877e460fe0d1d25398f35976b4d730/> `start_completing` should not be invoked on a per-iteration basis
   There's still some UB that can be triggered due to llama.cpp's threading model, which needs patching up.
 - <csr-id-81e5de901a3da88a97ba00c6a36e303d8708380d/> require `llama_context` is accessed from behind a mutex
   This solves a race condition when several `get_completions` threads are spawned at the same time
 - <csr-id-27706de1a471b317e4b7b4fdd4c5bbabfbd95ed6/> `start_completing` should not be invoked on a per-iteration basis
   There's still some UB that can be triggered due to llama.cpp's threading model, which needs patching up.

### Commit Statistics

<csr-read-only-do-not-edit/>

 - 11 commits contributed to the release.
 - 13 days passed between releases.
 - 8 commits were understood as [conventional](https://www.conventionalcommits.org).
 - 0 issues like '(#ID)' were seen in commit messages

### Commit Details

<csr-read-only-do-not-edit/>

<details><summary>view details</summary>

 * **Uncategorized**
    - Release llama_cpp_sys v0.2.1, llama_cpp v0.1.1 ([`ef4e3f7`](https://github.com/binedge/llama_cpp-rs/commit/ef4e3f7a3c868a892f26acfae2a5211de4900d1c))
    - Add `LlamaModel::load_from_file_async` ([`bbf9f69`](https://github.com/binedge/llama_cpp-rs/commit/bbf9f69a2dd068a3a20199ffce44d3c8a25b64d5))
    - Remove debug binary from Cargo.toml ([`dbdd9a4`](https://github.com/binedge/llama_cpp-rs/commit/dbdd9a4a2d813d990e5829a09fc5c8df75d9d54b))
    - Require `llama_context` is accessed from behind a mutex ([`81e5de9`](https://github.com/binedge/llama_cpp-rs/commit/81e5de901a3da88a97ba00c6a36e303d8708380d))
    - `start_completing` should not be invoked on a per-iteration basis ([`27706de`](https://github.com/binedge/llama_cpp-rs/commit/27706de1a471b317e4b7b4fdd4c5bbabfbd95ed6))
    - Update to llama.cpp 0a7c980 ([`eb8f627`](https://github.com/binedge/llama_cpp-rs/commit/eb8f62777aa63787004771d86d34a8862b3a4157))
    - Add `LlamaModel::load_from_file_async` ([`3bada65`](https://github.com/binedge/llama_cpp-rs/commit/3bada658c9139af1c3dcdb32c60c222efb87a9f6))
    - Remove debug binary from Cargo.toml ([`3eddbab`](https://github.com/binedge/llama_cpp-rs/commit/3eddbab3cc35a59acbe66fa4f5333a9ca0edb326))
    - Require `llama_context` is accessed from behind a mutex ([`b676baa`](https://github.com/binedge/llama_cpp-rs/commit/b676baa3c1a6863c7afd7a88b6f7e8ddd2a1b9bd))
    - `start_completing` should not be invoked on a per-iteration basis ([`4eb0bc9`](https://github.com/binedge/llama_cpp-rs/commit/4eb0bc9800877e460fe0d1d25398f35976b4d730))
    - Update to llama.cpp 0a7c980 ([`94d7385`](https://github.com/binedge/llama_cpp-rs/commit/94d7385fefdab42ac6949c6d47c5ed262db08365))
</details>

## v0.1.0 (2023-10-25)

<csr-id-702a6ff49d83b10a0573a5ca1fb419efaa43746e/>
<csr-id-116fe8c82fe2c43bf9041f6dbfe2ed15d00e18e9/>
<csr-id-96548c840d3101091c879648074fa0ed1cee3011/>
<csr-id-a5fb19499ecbb1060ca8211111f186efc6e9b114/>
<csr-id-aa5eed4dcb6f50b25c878e584787211402a9138b/>

### Chore

 - <csr-id-702a6ff49d83b10a0573a5ca1fb419efaa43746e/> remove `include` from llama_cpp
 - <csr-id-116fe8c82fe2c43bf9041f6dbfe2ed15d00e18e9/> Release
 - <csr-id-96548c840d3101091c879648074fa0ed1cee3011/> latest fixes from upstream

### Chore

 - <csr-id-aa5eed4dcb6f50b25c878e584787211402a9138b/> add CHANGELOG.md

### Bug Fixes

 - <csr-id-2cb06aea62b892a032f515b78d720acb915f4a22/> use SPDX license identifiers

### Other

 - <csr-id-a5fb19499ecbb1060ca8211111f186efc6e9b114/> configure for `cargo-release`

### Commit Statistics

<csr-read-only-do-not-edit/>

 - 9 commits contributed to the release over the course of 5 calendar days.
 - 6 commits were understood as [conventional](https://www.conventionalcommits.org).
 - 1 unique issue was worked on: [#3](https://github.com/binedge/llama_cpp-rs/issues/3)

### Commit Details

<csr-read-only-do-not-edit/>

<details><summary>view details</summary>

 * **[#3](https://github.com/binedge/llama_cpp-rs/issues/3)**
    - Release ([`116fe8c`](https://github.com/binedge/llama_cpp-rs/commit/116fe8c82fe2c43bf9041f6dbfe2ed15d00e18e9))
 * **Uncategorized**
    - Release llama_cpp v0.1.0 ([`f24c7fe`](https://github.com/binedge/llama_cpp-rs/commit/f24c7fe3ebd851a56301ce3d5a1b4250d2d797b9))
    - Add CHANGELOG.md ([`aa5eed4`](https://github.com/binedge/llama_cpp-rs/commit/aa5eed4dcb6f50b25c878e584787211402a9138b))
    - Remove `include` from llama_cpp ([`702a6ff`](https://github.com/binedge/llama_cpp-rs/commit/702a6ff49d83b10a0573a5ca1fb419efaa43746e))
    - Use SPDX license identifiers ([`2cb06ae`](https://github.com/binedge/llama_cpp-rs/commit/2cb06aea62b892a032f515b78d720acb915f4a22))
    - Release llama_cpp_sys v0.2.0 ([`d1868ac`](https://github.com/binedge/llama_cpp-rs/commit/d1868acd16a284b60630b4519af710f54fea3dca))
    - Latest fixes from upstream ([`96548c8`](https://github.com/binedge/llama_cpp-rs/commit/96548c840d3101091c879648074fa0ed1cee3011))
    - Configure for `cargo-release` ([`a5fb194`](https://github.com/binedge/llama_cpp-rs/commit/a5fb19499ecbb1060ca8211111f186efc6e9b114))
    - Initial commit ([`6f672ff`](https://github.com/binedge/llama_cpp-rs/commit/6f672ffddc49ce23cd3eb4996128fe8614c560b4))
</details>

