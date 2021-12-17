{
  description = "nnoir";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-21.11";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        py = pkgs.python38;
        customOverrides = self: super: {
          platformdirs = super.platformdirs.overridePythonAttrs (
            old: {
              postPatch = "";
            }
          );
          pytest = super.pytest.overridePythonAttrs (
            old: {
              postPatch = "";
            }
          );
          zipp = super.zipp.overridePythonAttrs (
            old: {
              prePatch = "";
            }
          );
          onnxruntime = super.onnxruntime.overridePythonAttrs (
            old: {
              nativeBuildInputs = [ ];
              postFixup =
                let rPath = pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc ];
                in
                ''
                  rrPath=${rPath}
                  find $out/lib -name '*.so' -exec patchelf --add-rpath "$rrPath" {} \;
                '';
            }
          );
        };
        nnoir = pkgs.poetry2nix.mkPoetryApplication {
          projectDir = ./nnoir;
          python = py;
          overrides = pkgs.poetry2nix.overrides.withDefaults (
            customOverrides
          );
          preferWheels = true;
        };
        nnoir-onnx = pkgs.poetry2nix.mkPoetryApplication {
          projectDir = ./nnoir-onnx;
          python = py;
          overrides = pkgs.poetry2nix.overrides.withDefaults (
            customOverrides
          );
          preferWheels = true;
        };
      in
      {
        packages.nnoir = nnoir;
        packages.nnoir-onnx = nnoir-onnx;

        devShell = pkgs.mkShell {
          buildInputs = [
            pkgs.protobuf

            py
            py.pkgs.jedi-language-server
            py.pkgs.poetry
          ];

          shellHook = ''
            export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc ]}
          '';
        };
      }
    );
}
