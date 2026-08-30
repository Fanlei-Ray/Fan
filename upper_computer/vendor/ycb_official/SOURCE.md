# YCB official real-object assets

These files are texture-mapped scans from the official Yale-CMU-Berkeley
(YCB) Object and Model Set, used for robotic manipulation research.

- Source: https://www.ycbbenchmarks.com/object-models/
- Download host: https://ycb-benchmarks.s3.amazonaws.com/index.html
- Dataset license: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Model variant: Google 16k textured meshes
- Objects: `006_mustard_bottle`, `011_banana`, `013_apple`, `017_orange`, `025_mug`, `056_tennis_ball`

The untouched downloaded archives are retained under `archives/`. Extracted
meshes and texture maps are under `objects/`. MuJoCo uses the original visual
meshes with simple collision proxies so the simulation remains stable and fast.
