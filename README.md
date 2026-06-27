<p align="center">
  <img src="logo.png" width="128" alt="Vocal Magic" />
</p>

# vocal-magic

Vocal magic is a collection of tools based on deep learning to help you work on vocal music.

## Run

Download checkpoints and organized them into this directory structure:

```
checkpoints
├── GAME-1.0-large
│   ├── config.yaml
│   ├── GAME-1.0-large.safetensors
│   └── lang_map.json
├── mbr-win10-sink8
│   └── mbr-win10-sink8.mlx.safetensors
├── melband-roformer
│   └── melband-roformer.mlx.safetensors
└── renaissance
    └── smule-renaissance-small.safetensors
```

Then run the webapp
```sh
cd webapp
cargo run -- --checkpoints-dir ../checkpoints/
```