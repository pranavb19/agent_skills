# Tool Selection

Pick by job. Depth-of-need decides which reference to read next. All of these are JS/TS, mostly free, and (for components) converging on the shadcn copy-paste registry model.

## Expanded decision table

| Use case | Primary | Why | Alternatives | Read |
|---|---|---|---|---|
| Scroll storytelling, pinning, scrubbed timelines | **GSAP + ScrollTrigger** | Most powerful & battle-tested; now 100% free incl. all plugins; framework-agnostic | Motion `useScroll`; native CSS scroll-driven animations | `tool-gsap-scrolltrigger.md` |
| React UI animation (gestures, layout, presence) | **Motion** (`motion/react`) | Declarative, springs, layout/FLIP, AnimatePresence; tiny footprint | GSAP (imperative); CSS transitions for trivial cases | `tool-motion.md` |
| 3D inside React | **R3F + Drei** | Declarative Three.js in JSX; huge helper ecosystem | Vanilla Three.js embedded; Spline | `tool-react-three-fiber.md` |
| 3D without React / max control | **Three.js (vanilla)** | Full control, no React overhead, works anywhere | R3F; Babylon.js | `tool-threejs.md` |
| No-code / designer-built 3D | **Spline** | Browser editor → web/React runtime; fastest idea→embed | Blender→GLTF→R3F; hand-coded Three.js | `tool-spline.md` |
| Silky smooth scroll | **Lenis** | <4kb, wraps native scroll (sticky/anchors/a11y survive), syncs with GSAP/WebGL | native `scroll-behavior`; GSAP ScrollSmoother | `tool-lenis.md` |
| Landing-page "wow" effects | **Aceternity / Magic UI** | Copy-paste animated components on Tailwind + Motion via shadcn CLI | Hand-built with Motion/GSAP for uniqueness | `tool-component-libraries.md` |
| App design-system foundation | **shadcn/ui** | Own-the-code Radix/Base UI + Tailwind; accessible, themable | Radix Themes; Park UI; Mantine | `tool-shadcn-ui.md` |
| Smooth scroll + 3D synced to DOM | **Lenis + R3F** (or `@14islands/r3f-scroll-rig`) | One RAF loop drives both | Drei `ScrollControls` (self-contained, no DOM scroll) | `combining-tools.md` |

## Selection heuristics

- **Imperative, complex, sequenced, cross-browser timeline?** GSAP. **Declarative React state-driven motion, gestures, layout transitions?** Motion. They coexist fine in one app — use each for its strength.
- **Need React to own the 3D scene graph / drive it from state?** R3F. **Standalone, embedding into non-React, or want zero abstraction?** vanilla Three.js. **A designer should build it without code?** Spline.
- **Want premium scroll feel site-wide?** Lenis (wire into GSAP's ticker). **Just need a scrubbed hero?** GSAP ScrollTrigger alone may be enough.
- **Building an app / dashboard / product UI?** shadcn/ui foundation. **Building a flashy marketing landing page fast?** shadcn/ui + selectively restyled Aceternity/Magic UI.

## Ecosystem facts to keep current (verify at install)

- **GSAP** is now 100% free including all former Club plugins (SplitText, MorphSVG, ScrollSmoother, etc.) and commercial use, since v3.13 (April 2025), after Webflow acquired GreenSock (Oct 2024).
- **Framer Motion → Motion**: package is `motion`, React import is `motion/react`; gained a vanilla/Vue API and a hybrid (hardware-accelerated) engine. Migration is mostly an import swap.
- **shadcn / Aceternity / Magic UI** all install through the `shadcn` CLI and support **namespaced registries** (`@aceternity/...`, `@magicui/...`), so they compose cleanly and you own the code.
- **R3F** version pairing: v9 ↔ React 19, v8 ↔ React 18.
- **Three.js** r130+ needs an import map (or a bundler) for browser ES modules.
- **Lenis** packages renamed from `@studio-freight/*` to `lenis` and `lenis/react`.
