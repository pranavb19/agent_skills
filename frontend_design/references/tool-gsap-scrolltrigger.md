# Tool: GSAP + ScrollTrigger

For scroll storytelling, pinning, scrubbed timelines, and complex sequenced animation. The most powerful, battle-tested option, now fully free.

## Table of contents
- Licensing (verify-first)
- Core concepts
- ScrollTrigger essentials
- React integration (the right way)
- Best practices and pitfalls
- Useful plugins

## Licensing (verify-first)

With the **v3.13 release (April 30, 2025)**, GSAP is **100% free for everyone, including all previously paid "Club" plugins** (SplitText, MorphSVG, ScrollSmoother, DrawSVG, Inertia, etc.) and **commercial use**. This followed Webflow's acquisition of GreenSock in October 2024, with Webflow now hosting it. Practically: `npm i gsap` and register any plugin — no token, no paywall. Don't choose lighter libraries on cost grounds anymore.

## Core concepts

- **Tween** — one animation: `gsap.to(target, vars)`, `gsap.from`, `gsap.fromTo`, `gsap.set` (no animation, just apply).
- **Timeline** — a sequence with relative position labels: `"<"` (start with previous), `">"` (after previous), `"<+0.5"`, `"-=0.2"`, named labels.
- **Easing** — `power1`–`power4`, `back`, `elastic`, `expo`, `sine`, `steps()`, plus `CustomEase`. Easing is the single most-felt quality of motion.

```js
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
gsap.registerPlugin(ScrollTrigger);

const tl = gsap.timeline({
  scrollTrigger: {
    trigger: ".scene",
    start: "top top",      // "triggerPos viewportPos"
    end: "+=2000",         // 2000px of scroll distance
    scrub: 1,              // bind progress to scroll; number = seconds of smoothing
    pin: true,             // lock .scene while active
    snap: { snapTo: "labels", duration: 0.3, ease: "power1.inOut" },
    markers: false,        // true while developing
  },
});
tl.to(".title", { y: -100, opacity: 0 })
  .to(".image", { scale: 1.4 }, "<")
  .addLabel("mid")
  .to(".caption", { x: 0, opacity: 1 });
```

## ScrollTrigger essentials

- **`scrub`** binds animation progress to the scrollbar; a number adds that many seconds of "catch-up" smoothing (`true` = instant). With scrub, `end` controls *how far you scroll*, not playback speed.
- **`pin`** locks the trigger (or `pin: ".selector"`). ScrollTrigger wraps the pinned element in a `.pin-spacer` and adds padding so following content doesn't collapse; use `pinSpacing: false` to disable (it's effectively off inside flex containers).
- **`snap`** — `true`, an array of progress values, `"labels"`, `"labelsDirectional"`, or a function; supports `delay`, `duration`, `ease`, `inertia`.
- **`start` / `end`** — `"triggerPos viewportPos"` syntax: `"top 80%"`, `"center center"`, `"bottom top"`, or `"+=1500"`.
- **`toggleActions`** — four space-separated actions for onEnter/onLeave/onEnterBack/onLeaveBack (`play pause resume reset restart complete reverse none`). Use for non-scrubbed play-on-enter animations.
- **`markers: true`** — indispensable while building; remove for production.
- **`containerAnimation`** — for horizontal sections: tie nested triggers to a horizontal master tween.
- **`gsap.matchMedia()` / `ScrollTrigger.matchMedia()`** — responsive: simplify or disable heavy triggers on small screens.

Horizontal scroll skeleton:
```js
const sections = gsap.utils.toArray(".panel");
gsap.to(sections, {
  xPercent: -100 * (sections.length - 1),
  ease: "none",
  scrollTrigger: { trigger: ".track", pin: true, scrub: 1,
    end: () => "+=" + document.querySelector(".track").offsetWidth },
});
```

## React integration (the right way)

Use `@gsap/react`'s `useGSAP()` — a drop-in for `useEffect`/`useLayoutEffect` that runs cleanup automatically via `gsap.context()`. This is essential because React 18+ Strict Mode double-invokes effects; without `context().revert()` you get duplicated tweens and ScrollTriggers leaking on detached nodes.

```jsx
"use client"; // required in the App Router
import { useRef } from "react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { useGSAP } from "@gsap/react";
gsap.registerPlugin(ScrollTrigger, useGSAP);

export function Scene() {
  const container = useRef(null);
  const { contextSafe } = useGSAP(() => {
    gsap.from(".item", {
      opacity: 0, y: 80, stagger: 0.1,
      scrollTrigger: { trigger: container.current, start: "top 80%" },
    });
  }, { scope: container }); // selector text is scoped & auto-reverted on unmount

  // animations created AFTER mount (event handlers) must be contextSafe:
  const onClick = contextSafe(() => gsap.to(".item", { rotation: 180 }));
  return <div ref={container}>{/* .item children */}</div>;
}
```

`useGSAP` is SSR-safe (isomorphic layout effect) but the component must be a client component.

## Best practices and pitfalls

- Animate **transforms/opacity**, not layout properties.
- **Don't animate the pinned element itself** — animate its children.
- Call **`ScrollTrigger.refresh()`** after dynamic content/height changes (fonts, images, route transitions).
- Define image/element **aspect ratios in CSS** so document height is stable (prevents jumpy triggers and CLS).
- **Kill triggers on unmount** (`useGSAP`/`context` does this).
- Go easy on `scrub` smoothing and pinning on mobile; gate heavy effects with `matchMedia`.
- With Lenis, share GSAP's ticker — never run two RAF loops (see `combining-tools.md`).

## Useful plugins (all free now)

- **ScrollTrigger** — scroll-driven everything.
- **ScrollSmoother** — GSAP's own smooth scroll (alternative to Lenis; built on ScrollTrigger).
- **SplitText** — split into chars/words/lines for staggered text reveals.
- **MorphSVG** — morph one path into another.
- **DrawSVG** — animate stroke drawing.
- **Flip** — FLIP layout transitions.
- **Observer** — unified wheel/touch/pointer events.
