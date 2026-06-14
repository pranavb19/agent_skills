# Craft: Performance

Performance is craft. A beautiful page that blocks for three seconds or janks on interaction is slop with better paint. Treat every animation/3D library as a budget line item.

## Table of contents
- Core Web Vitals (2025–2026 thresholds)
- LCP
- INP (the hard one)
- CLS
- Budgeting animation/3D libraries
- GPU-friendly animation
- Monitoring

## Core Web Vitals (2025–2026 thresholds, 75th percentile)

- **LCP (Largest Contentful Paint)** < 2.5s — loading.
- **INP (Interaction to Next Paint)** < 200ms — responsiveness. INP officially replaced FID in March 2024 and is the most commonly failed Core Web Vital — a large share of sites still fail the 200ms threshold.
- **CLS (Cumulative Layout Shift)** < 0.1 — visual stability.

Set internal alerts at ~80% of thresholds (INP > 160ms, LCP > 2.0s, CLS > 0.08) so regressions are caught early.

## LCP

- Identify the LCP element (usually the hero image or headline) and make it fast: modern formats (AVIF/WebP), correct sizing, `fetchpriority="high"`, preload the hero asset.
- SSR/SSG the above-the-fold content.
- Eliminate render-blocking CSS/JS; inline critical CSS if needed.
- **Never let an animation or 3D bundle be in the critical path of LCP.** Lazy-load it after first paint.

## INP (the hard one)

INP is dominated by main-thread JavaScript. To keep it under 200ms:
- **Code-split** and defer non-critical JS; ship less to start.
- **Break up long tasks** (keep individual tasks < 50ms); `await` a yield (`scheduler.yield()` or `setTimeout(0)`) inside heavy loops.
- **Offload heavy compute** (3D physics, particle systems, parsing) to a **Web Worker** so the main thread stays free to paint interactions.
- Debounce/throttle expensive handlers; avoid layout thrash (batch reads then writes).
- For animation specifically: prefer compositor-only properties so the main thread isn't doing layout during interaction.

## CLS

- Set explicit `width`/`height` or `aspect-ratio` on images, video, iframes, and ad/embed slots so they reserve space before loading.
- Reserve space for async/injected content (banners, cookie bars).
- Use `font-display: optional` or `swap` and preload fonts; fluid `clamp()` type avoids breakpoint-jump shifts.
- Never insert content above existing content after load.

## Budgeting animation/3D libraries

These are not a false tradeoff with performance — combine beauty and speed via three techniques:
1. **Progressive loading:** load heavy interaction *after* critical content paints.
2. **Code-splitting:** dynamically import animation/3D code when it becomes visible or on interaction.
3. **Main-thread offloading:** Web Workers for heavy compute.

Concrete budgets and tactics:
- **R3F / Three.js / Spline canvases:** lazy-load with `next/dynamic` (`ssr: false`) so WebGL never blocks LCP. A 3D scene can add 2–5MB — it must never be in the initial bundle. Use on-demand rendering (`frameloop="demand"` in R3F) for mostly-static scenes; compress geometry (Draco/Meshopt) and textures (≤1024² where possible). Ship a static image/video fallback for low-end devices and reduced-motion users. Target ≥30fps on mid-range mobile or degrade gracefully.
- **Motion:** use `LazyMotion` + `m` components to shrink the bundle (~4.6KB vs full); virtualize lists over ~50 animated items.
- **GSAP:** register only the plugins you use; gate heavy ScrollTriggers on viewport with `matchMedia` so mobile does less.
- **Component libraries (Aceternity/Magic UI):** lazy-load heavy animated backgrounds; they're all client components.

## GPU-friendly animation

- Animate **`transform`** and **`opacity`** only for smooth 60fps; these are composited off the main thread and skip layout/paint. Avoid animating `width`, `height`, `top`, `left`, `margin`, `box-shadow` (use a layered pseudo-element trick for shadow if needed).
- Use `will-change` **surgically** (only just before an animation, remove after) — it allocates a layer and costs memory if left on many elements.
- `content-visibility: auto` and CSS `contain` reduce off-screen layout/paint cost.

## Monitoring

- Measure field data with the `web-vitals` library (`onINP`, `onLCP`, `onCLS`) and send to analytics.
- Run **Lighthouse CI** in the pipeline and block deploys on regressions.
- Use real-user monitoring (CrUX / your RUM) — lab scores miss real-device INP. Test on a throttled mid-range Android, not just your laptop.
