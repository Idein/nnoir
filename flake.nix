{
  description = "nnoir";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    poetry2nix = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };
  outputs = { self, nixpkgs, flake-utils, poetry2nix }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        inherit (poetry2nix.lib.mkPoetry2Nix { inherit pkgs; }) mkPoetryApplication overrides;
        pkgs = nixpkgs.legacyPackages.${system};
        py = pkgs.python39;
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
        nnoir = mkPoetryApplication {
          projectDir = ./nnoir;
          python = py;
          overrides = overrides.withDefaults (
            customOverrides
          );
          preferWheels = true;
        };
        nnoir-onnx = mkPoetryApplication {
          projectDir = ./nnoir-onnx;
          python = py;
          overrides = overrides.withDefaults (
            customOverrides
          );
          preferWheels = true;
        };
      in
      {
        packages.nnoir = nnoir;
        packages.nnoir-onnx = nnoir-onnx;

        packages.dockerimage = pkgs.dockerTools.buildImage {
          name = "idein/nnoir-tools";
          tag = "latest";
          created = "now";
          copyToRoot = [
            nnoir
            nnoir-onnx
          ];
          config = {
            WorkingDir = "/work";
            Env = [ "PATH=/bin:$PATH" ];
          };
        };

        devShell = pkgs.mkShell {
          buildInputs = [
            pkgs.protobuf

            py
            pkgs.poetry
          ];

          shellHook = ''
            export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc ]}
          '';
        };
      }
    );
}
