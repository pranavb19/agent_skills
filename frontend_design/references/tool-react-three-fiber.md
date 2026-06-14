# Tool: React Three Fiber + Drei

For 3D scenes inside React. R3F is a React renderer for Three.js — `<mesh/>` becomes `new THREE.Mesh()` — so everything in Three.js works, and React's scheduler can make it perform well at scale.

## Table of contents
- Version pairing (critical)
- Canvas and basic scene
- Key hooks
- Essential Drei helpers
- Loading models
- Scroll inside R3F
- Performance (where 3D lives or dies)

## Version pairing (critical)

- `@react-three/fiber@9` ↔ **React 19**
- `@react-three/fiber@8` ↔ **React 18**
- `@react-three/drei` is the helper library (10.x line).

Install: `npm i three @react-three/fiber @react-three/drei`.

## Canvas and basic scene

```jsx
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Environment, useGLTF } from "@react-three/drei";
import { Suspense, useRef } from "react";

function Model(props) {
  const { scene } = useGLTF("/model.glb");
  const ref = useRef();
  useFrame((state, delta) => { ref.current.rotation.y += delta * 0.2; });
  return <primitive ref={ref} object={scene} {...props} />;
}

export default function Scene() {
  return (
    <Canvas camera={{ position: [0, 0, 5], fov: 45 }} frameloop="demand" dpr={[1, 2]}>
      <ambientLight intensity={Math.PI / 2} />
      <spotLight position={[10, 10, 10]} angle={0.15} penumbra={1} />
      <Suspense fallback={null}>
        <Model />
        <Environment preset="city" />   {/* image-based lighting + reflections */}
      </Suspense>
      <OrbitControls enableDamping />
    </Canvas>
  );
}
useGLTF.preload("/model.glb");
```

JSX maps to Three.js: `<mesh>` + `<boxGeometry args={[1,1,1]} />` + `<meshStandardMaterial color="hotpink" />`. `args` are constructor arguments; nested props set properties (`position={[x,y,z]}`, `rotation`, `scale`).

## Key hooks

- **`useFrame((state, delta) => {})`** — runs every rendered frame; `delta` is seconds since last frame (multiply movement by it for frame-rate independence). Don't allocate inside it.
- **`useThree()`** — exposes `camera`, `gl` (renderer), `scene`, `size`, `viewport`, `invalidate`.
- **`useLoader(GLTFLoader, url)`** — Suspense-based asset loading (Drei's `useGLTF` wraps this with Draco support and caching).

## Essential Drei helpers

`OrbitControls`, `CameraControls`, `Environment` (HDRI lighting/reflections), `useGLTF`/`Gltf`, `Html` (render DOM in 3D space), `Text`/`Text3D`, `Float`, `ContactShadows`, `MeshTransmissionMaterial` (glass), `Bounds` (auto-fit camera), `Instances`/`Merged` (instancing), `ScrollControls`/`Scroll`/`useScroll`, and `<Perf/>` from `r3f-perf` for FPS/draw-calls/memory.

## Loading models

- Author/optimize in Blender → export **glTF/GLB**. Compress with **Draco** or **Meshopt**.
- `npx gltfjsx model.glb` generates a typed React component from a model (lets you target individual meshes/materials and animate them).
- Always wrap loaders in `<Suspense>` and `preload` critical assets.

## Scroll inside R3F

Drei's **`ScrollControls`** creates an invisible scroll container (no real DOM scroll); **`useScroll()`** gives a dampened `offset` (0–1) plus `range`/`curve`/`visible` helpers. A common pattern drives a paused GSAP timeline by scroll offset:

```jsx
<ScrollControls pages={3} damping={0.25}>
  <Office />  {/* uses useScroll + a paused gsap.timeline */}
  <Scroll html><h1 style={{ top: "100vh" }}>Section 2</h1></Scroll>
</ScrollControls>
```
```jsx
const scroll = useScroll();
useFrame(() => tl.current.seek(scroll.offset * tl.current.duration()));
```

To sync WebGL to *real* page scroll across a whole site (mixing DOM and 3D), use `@14islands/r3f-scroll-rig` (`<GlobalCanvas>` + `<SmoothScrollbar>`, then `<UseCanvas>` + `<ScrollScene track={ref}>`), which is purpose-built and respects on-demand rendering. See `combining-tools.md`.

## Performance (where 3D lives or dies)

- **On-demand rendering:** `frameloop="demand"` renders only when something changes; call `invalidate()` (from `useThree`) to request a frame. Huge CPU/battery savings for mostly-static scenes.
- **Instancing** for many repeated meshes (`<Instances>` / `InstancedMesh`) — one draw call instead of thousands.
- **Lazy-load the Canvas** (`next/dynamic`, `ssr: false`) so WebGL never blocks LCP, and ship a static fallback for low-end/reduced-motion.
- **Cap DPR** (`dpr={[1, 2]}`) and compress textures (≤1024² where possible).
- **Dispose** geometries/materials/textures you remove; reuse materials.
- **Profile** with Drei/`r3f-perf` `<Perf/>`; watch draw calls and triangle count.
- Target ≥30fps on mid-range mobile or degrade gracefully. See `craft-performance.md` for the budget.
