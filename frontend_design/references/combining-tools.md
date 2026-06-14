# Combining Tools Into a Cohesive Experience

Award-tier experiences are combinations. The gotchas live at the seams — competing RAF loops, double renders, SSR, and cleanup. This is where most integrations break.

## Table of contents
- Lenis + GSAP ScrollTrigger
- R3F + scroll
- Motion + shadcn/ui
- Aceternity/Magic UI + shadcn
- The universal gotcha: cleanup & SSR
- A cohesive recipe
- Staged build order

## Lenis + GSAP ScrollTrigger

Drive both from **one** loop. Do NOT run a separate `requestAnimationFrame` for Lenis at the same time as GSAP's ticker.
```js
const lenis = new Lenis();
lenis.on("scroll", ScrollTrigger.update);
gsap.ticker.add((time) => lenis.raf(time * 1000));
gsap.ticker.lagSmoothing(0);
```
Call `ScrollTrigger.refresh()` after content/height changes. On mobile, test `syncTouch` tradeoffs (smoothness vs FPS and scroll-focus jumps). Honor reduced motion by disabling Lenis smoothing.

## R3F + scroll

Two valid approaches:
1. **Self-contained canvas** — Drei `ScrollControls` + `useScroll` (no real DOM scroll). Drive a paused GSAP timeline by `scroll.offset`. Good for a contained 3D section.
2. **Synced to real page scroll** — `@14islands/r3f-scroll-rig` (`<GlobalCanvas>` + `<SmoothScrollbar>`, then `<UseCanvas>` + `<ScrollScene track={ref}>`) or share Lenis's loop, so WebGL tracks DOM elements across a whole site.

Gotchas: if you combine global postprocessing with viewport scenes, disable the global render (`globalRender={false}`) to avoid double renders. Keep `frameloop="demand"` and call `invalidate()` when scroll changes something so you're not rendering idle frames.

## Motion + shadcn/ui

Tailwind/Radix handle structure and accessibility; Motion handles movement.
- Wrap or compose shadcn primitives with `motion.*`.
- Use `AnimatePresence` for dialogs/sheets — keep `<AnimatePresence>` mounted, give children unique keys.
- Wrap the app in `<MotionConfig reducedMotion="user">`.
- **Don't unmount focus-trapped content mid-animation** — let Radix manage focus; animate the content, not the focus lifecycle.

## Aceternity/Magic UI + shadcn

They share the registry and token system, so they compose cleanly. But **restyle to your tokens** to avoid the template look, **lazy-load** heavy animated backgrounds, and remember they're all client components. Pick one signature effect; keep the rest restrained.

## The universal gotcha: cleanup & SSR

- **Cleanup (React):** GSAP `useGSAP()`/`gsap.context().revert()`; kill ScrollTriggers; `lenis.destroy()`; dispose Three.js geometries/materials/textures; stable keys for `AnimatePresence`.
- **SSR / first paint:** mark animation/3D components `"use client"` and lazy-load them (`next/dynamic`, `ssr: false`) so they never block LCP. Provide static fallbacks.
- **One RAF loop:** when multiple libraries animate per frame, route them through a single ticker (usually GSAP's) instead of several independent loops.

## A cohesive recipe

1. **Foundation:** shadcn/ui + your OKLCH token system + distinctive type (`tool-shadcn-ui.md`, `craft-color.md`, `craft-typography.md`).
2. **Feel:** Lenis for smooth scroll, wired into GSAP's ticker.
3. **Narrative:** GSAP + ScrollTrigger for the scroll story (pin/scrub), via `useGSAP`.
4. **Anchor:** ONE 3D moment — lazy R3F (`frameloop="demand"`) or a Spline scene — as the single visual centerpiece, not 3D everywhere.
5. **Micro-interactions:** Motion for in-app gestures, layout transitions, presence.
6. **Guardrails:** everything budgeted against LCP/INP/CLS and `prefers-reduced-motion` (`craft-performance.md`, `craft-accessibility.md`, `craft-motion.md`).

## Staged build order

- **Stage 1 — Foundation:** scaffold Next.js + Tailwind + shadcn/ui; define type scale, spacing scale, OKLCH tokens, and motion tokens *before* building. Replace default fonts and palette immediately. Goal: a static page that already feels intentional with zero animation.
- **Stage 2 — Motion:** add Motion micro-interactions and `whileInView` reveals; add Lenis if you want premium feel. Apply the frequency gate (rare = expressive, frequent/keyboard = instant). Goal: INP stays <200ms; reduced-motion path verified.
- **Stage 3 — Scroll story:** GSAP + ScrollTrigger via `useGSAP`; animate children of pinned elements; `refresh()` on dynamic height. Goal: no CLS from pinning; smooth on mid-range Android.
- **Stage 4 — 3D (only if it earns its place):** Spline for fast designer-built, or R3F/Three.js for control. Lazy-load, on-demand render, compress assets, static fallback. Goal: 3D adds nothing to LCP; ≥30fps mid-range mobile or degrade.
- **Stage 5 — Polish & guardrails:** add `web-vitals` reporting + Lighthouse CI (block regressions). Run the anti-slop audit (`craft-anti-slop.md`). If INP >200ms or LCP >2.5s, cut/defer animation before adding more; if a 3D scene can't hit 30fps on mid-range mobile, replace with static/video; if it looks templated, restyle tokens or hand-build the signature effect.
