# Tool: Three.js (Vanilla)

For 3D without React, maximum control, or embedding into an existing imperative codebase. Current line is **r180+** (npm `three@0.18x`).

## Table of contents
- Module setup (import maps)
- Fundamentals
- Minimal scene
- Materials and lights
- Loading models and post-processing
- Shaders
- Performance

## Module setup (import maps)

Since r130, browser ES-module usage requires an **import map** (or a bundler like Vite). Without a bundler:

```html
<script type="importmap">
{ "imports": {
  "three": "https://unpkg.com/[email protected]/build/three.module.js",
  "three/addons/": "https://unpkg.com/[email protected]/examples/jsm/"
}}
</script>
```
With a bundler, just `npm i three` and `import * as THREE from "three"`; addons live under `three/examples/jsm/...` or the `three/addons/` alias.

## Fundamentals

Scene (object graph) → Camera (PerspectiveCamera for realism, OrthographicCamera for flat/iso) → Renderer (WebGLRenderer). You add meshes (geometry + material) and lights to the scene, then render each frame.

## Minimal scene

```html
<script type="module">
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, innerWidth/innerHeight, 0.1, 100);
camera.position.set(0, 1, 5);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(Math.min(devicePixelRatio, 2)); // cap DPR for perf
document.body.appendChild(renderer.domElement);

scene.add(new THREE.AmbientLight(0xffffff, 0.6));
const key = new THREE.DirectionalLight(0xffffff, 2); key.position.set(5, 5, 5);
scene.add(key);

const mesh = new THREE.Mesh(
  new THREE.IcosahedronGeometry(1, 0),
  new THREE.MeshStandardMaterial({ color: 0x6699ff, roughness: 0.3, metalness: 0.1 })
);
scene.add(mesh);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

renderer.setAnimationLoop(() => {        // preferred over manual rAF (XR-safe)
  mesh.rotation.y += 0.005;
  controls.update();
  renderer.render(scene, camera);
});
addEventListener("resize", () => {
  camera.aspect = innerWidth/innerHeight; camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});
</script>
```

## Materials and lights

- **Geometries:** Box, Sphere, Plane, Icosahedron, Torus, and `BufferGeometry` for custom.
- **Materials:** `MeshBasicMaterial` (unlit), `MeshStandardMaterial` / `MeshPhysicalMaterial` (PBR — roughness, metalness, clearcoat, transmission for glass), `ShaderMaterial` (custom GLSL).
- **Lights:** Ambient, Directional (sun), Point, Spot, plus image-based lighting via an environment map (`scene.environment`) — IBL usually does more for realism than stacking lights. Use a HDRI/`RoomEnvironment` for instant quality.
- Set `renderer.toneMapping = THREE.ACESFilmicToneMapping` and correct color space for natural results.

## Loading models and post-processing

```js
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";
const draco = new DRACOLoader(); draco.setDecoderPath("/draco/");
const loader = new GLTFLoader(); loader.setDRACOLoader(draco);
loader.load("/model.glb", (gltf) => scene.add(gltf.scene));
```

Post-processing via `EffectComposer` + passes (`RenderPass`, `UnrealBloomPass`, etc.) from `three/addons/postprocessing/`.

## Shaders

Custom visual effects use `ShaderMaterial` with `uniforms`, a `vertexShader`, and a `fragmentShader` (GLSL). Drive uniforms (e.g. `uTime`) from the animation loop for animated effects. This is the route to bespoke, non-generic visuals (distortion, gradients, particle systems) — but budget the complexity.

## Performance

- Cap `pixelRatio` (≤2), merge geometry or use `InstancedMesh` for repeats, keep draw calls low.
- **Dispose** geometries/materials/textures you remove (`.dispose()`); reuse materials and textures.
- Prefer `setAnimationLoop` over manual `requestAnimationFrame` (handles WebXR and tab visibility better).
- Compress textures (≤1024² where possible) and geometry (Draco/Meshopt).
- Lazy-load the whole 3D module so it never blocks first paint; ship a static fallback. See `craft-performance.md`.
