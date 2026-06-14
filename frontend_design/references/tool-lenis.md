# Tool: Lenis (Smooth Scroll)

For a silky, premium smooth-scroll feel. Lenis (by Darkroom Engineering) is lightweight (<4kb), dependency-free, and — crucially — **wraps native scroll** rather than hijacking it, so `position: sticky`, anchor links, and accessibility keep working.

## Table of contents
- Why Lenis (and the package rename)
- Vanilla setup
- GSAP ScrollTrigger integration (canonical)
- React
- Options worth knowing
- Accessibility caveats

## Why Lenis (and the package rename)

It intercepts wheel/touch and drives scroll position via lerp, producing smoothness without breaking native behavior — the main advantage over scroll-hijacking rewrites. **Package rename:** the old `@studio-freight/lenis` and `@studio-freight/react-lenis` are deprecated — use **`lenis`** and **`lenis/react`**. The old `smoothTouch` option is gone (use `syncTouch`). Import the required CSS (`import "lenis/dist/lenis.css"`).

## Vanilla setup

```js
import Lenis from "lenis";
import "lenis/dist/lenis.css";

const lenis = new Lenis({ autoRaf: true }); // autoRaf drives its own RAF loop
lenis.on("scroll", (e) => { /* e.scroll, e.velocity, e.direction */ });
```

When integrating with another animation loop (GSAP, R3F), turn `autoRaf` off and drive `lenis.raf()` from the shared loop instead — never run two competing RAF loops.

## GSAP ScrollTrigger integration (canonical)

One shared ticker drives both:
```js
import Lenis from "lenis";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
gsap.registerPlugin(ScrollTrigger);

const lenis = new Lenis();
lenis.on("scroll", ScrollTrigger.update);          // keep triggers in sync
gsap.ticker.add((time) => lenis.raf(time * 1000)); // GSAP time is seconds → ms
gsap.ticker.lagSmoothing(0);
```

## React

Use the `lenis/react` wrapper:
```jsx
import { ReactLenis, useLenis } from "lenis/react";

export default function Layout({ children }) {
  return <ReactLenis root>{children}</ReactLenis>;
}
```
For GSAP integration in React, grab the instance via a ref and call `lenisRef.current?.lenis?.raf(time * 1000)` inside `gsap.ticker.add`. In Next.js App Router, this is a client component.

## Options worth knowing

- **`lerp`** (0–1) intensity, or **`duration`** + **`easing`** for the smoothing curve.
- **`orientation`** — `vertical` | `horizontal`.
- **`syncTouch`** — needed for some touch/infinite cases, but can hurt the "smooth" feel and cost FPS on mobile; test it.
- **`anchors`** — smooth anchor-link scrolling.
- **`prevent`** — a function to skip smoothing in certain conditions (e.g. when a modifier key is held).
- **`allowNestedScroll`** and **`data-lenis-prevent`** on a nested scroller — keep inner scroll areas (modals, code blocks) working natively.

## Accessibility caveats

Because Lenis runs on native scroll, it preserves keyboard scrolling and a11y better than hijack libraries — but smooth scrolling still alters expected behavior. Honor **`prefers-reduced-motion`** (disable or shorten smoothing), ensure focus-driven navigation still jumps correctly, and test keyboard `Page Down`/`Home`/`End`. On mobile, consider disabling smoothing if it costs frames. Add `data-lenis-prevent` to any inner scrollable region so it isn't smoothed. See `combining-tools.md` for syncing with GSAP/R3F.
