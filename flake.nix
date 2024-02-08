{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-23.11";
    systems.url = "github:nix-systems/default";
    flake-utils = {
      url = "github:numtide/flake-utils";
      inputs = { systems.follows = "systems"; };
    };
    devenv = {
      url = "github:cachix/devenv";
      inputs = { nixpkgs.follows = "nixpkgs"; };
    };
    fenix = {
      url = "github:nix-community/fenix";
      inputs = { nixpkgs.follows = "nixpkgs"; };
    };
  };

  nixConfig = {
    extra-trusted-public-keys =
      "devenv.cachix.org-1:w1cLUi8dv3hnoSPGAuibQv+f9TZLr6cv/Hm9XgU50cw=";
    extra-substituters = "https://devenv.cachix.org";
  };

  outputs = { self, nixpkgs, devenv, systems, flake-utils, fenix, ... }@inputs:
    with builtins; flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        rustToolchain = with fenix.packages.${system};
          combine [
            stable.rustc
            stable.cargo
            stable.rust-src
            stable.rust-std
            stable.rustfmt
            stable.rust-analyzer
            stable.clippy
            targets.x86_64-unknown-linux-musl.stable.rust-std
            targets.wasm32-unknown-unknown.stable.rust-std
          ];

        llvmPackages = pkgs.llvmPackages_11;

        clangBuildInputs = with llvmPackages; [
          clang
          libclang
          libcxx
          libcxxabi
          lld
          lldb
        ];

        nativeBuildInputs = with pkgs; clangBuildInputs ++ [
          rustToolchain
          ninja
          cmake
          gnumake
          pkg-config
        ];

        devInputs = clangBuildInputs ++ nativeBuildInputs ++ (with pkgs; [ nixfmt openssl vulkan-loader vulkan-headers ]);

        stdenv = pkgs.stdenv;
        lib = pkgs.lib;
      in {
        devShells.default = devenv.lib.mkShell {
          inherit inputs pkgs;

          modules = [{
            packages = devInputs;

            enterShell = ''
              export LD_LIBRARY_PATH=${
                pkgs.lib.makeLibraryPath devInputs
              }:$LD_LIBRARY_PATH

              export CPATH=${llvmPackages.libclang.lib}/lib/clang/${llvmPackages.libclang.version}/include:$CPATH

              export BINDGEN_EXTRA_CLANG_ARGS="$(< ${stdenv.cc}/nix-support/libc-crt1-cflags) \
                $(< ${stdenv.cc}/nix-support/libc-cflags) \
                $(< ${stdenv.cc}/nix-support/cc-cflags) \
                $(< ${stdenv.cc}/nix-support/libcxx-cxxflags) \
                ${lib.optionalString stdenv.cc.isClang "-idirafter ${stdenv.cc.cc}/lib/clang/${lib.getVersion stdenv.cc.cc}/include"} \
                ${lib.optionalString stdenv.cc.isGNU "-isystem ${stdenv.cc.cc}/include/c++/${lib.getVersion stdenv.cc.cc} -isystem ${stdenv.cc.cc}/include/c++/${lib.getVersion stdenv.cc.cc}/${stdenv.hostPlatform.config}"}
              "

              export VULKAN_SDK=${pkgs.vulkan-loader}
            '';
          }];
        };
      });
}
