# Tool: Motion (formerly Framer Motion)

For declarative React UI animation: gestures, layout/shared-element transitions, enter/exit presence, spring physics, and scroll-linked motion.

## Table of contents
- The rebrand
- Two flavors
- Core API
- Layout and shared-element animation
- AnimatePresence
- Scroll-linked animation
- Performance and accessibility

## The rebrand

Framer Motion became the independent project **Motion** (motion.dev) in 2025. The npm package is now **`motion`** (the old `framer-motion` still works but is no longer actively developed), and the React import is **`motion/react`**. Migration is a near-pure import swap:

```diff
- import { motion, AnimatePresence } from "framer-motion";
+ import { motion, AnimatePresence } from "motion/react";
```

## Two flavors

- **Motion for React** — `motion/react`: the `motion.*` components, hooks, layout/FLIP engine, `AnimatePresence`.
- **Framework-agnostic Motion** — `import { animate, scroll, inView } from "motion"`: vanilla JS / Vue, leaning on native browser APIs (WAAPI / ScrollTimeline) where possible, which is why simple transforms are very small. GSAP remains more capable for complex sequencing; use Motion for declarative React state-driven motion.

## Core API

```jsx
import { motion, AnimatePresence } from "motion/react";

// Basic + gestures
<motion.div
  initial={{ opacity: 0, y: 20 }}
  animate={{ opacity: 1, y: 0 }}
  whileHover={{ scale: 1.05 }}
  whileTap={{ scale: 0.95 }}
  whileFocus={{ scale: 1.02 }}   // keyboard-friendly state
  transition={{ type: "spring", stiffness: 300, damping: 25 }}
/>

// Viewport reveal
<motion.section
  initial={{ opacity: 0, y: 40 }}
  whileInView={{ opacity: 1, y: 0 }}
  viewport={{ once: true, margin: "-100px" }}
/>

// Variants + stagger orchestration
const list = { visible: { transition: { staggerChildren: 0.08, when: "beforeChildren" } },
               hidden:  { transition: { when: "afterChildren" } } };
const item = { hidden: { opacity: 0, y: 20 }, visible: { opacity: 1, y: 0 } };
<motion.ul variants={list} initial="hidden" animate="visible">
  {items.map(i => <motion.li key={i.id} variants={item}>{i.text}</motion.li>)}
</motion.ul>
```

**Springs** (`type: "spring"` with `stiffness`, `damping`, `mass`) feel natural and handle interruption/retargeting — prefer them for gesture-driven and layout motion.

## Layout and shared-element animation

Add the **`layout`** prop and Motion animates between any two layouts using transforms (FLIP), correcting scale distortion automatically — great for reordering lists, expanding cards, resizing panels.

```jsx
<motion.div layout transition={{ type: "spring", stiffness: 400, damping: 30 }} />
```

**`layoutId`** does shared-element transitions (one element appears to morph into another across mounts) — e.g., a thumbnail expanding into a detail view. `LayoutGroup`, `layoutScroll`, and `layoutRoot` handle advanced grouping/scroll cases.

## AnimatePresence

Keeps exiting elements in the DOM long enough to run their `exit` animation.

```jsx
<AnimatePresence mode="wait">
  {open && (
    <motion.div key="panel"
      initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} />
  )}
</AnimatePresence>
```

Rules: keep `<AnimatePresence>` itself mounted (don't put it inside the condition); every child needs a unique, stable `key`. Modes: `sync` (default), `wait` (finish exit before enter — good for route/page transitions), `popLayout` (removed element leaves flow immediately so siblings reflow during exit; wrap custom children in `forwardRef`).

## Scroll-linked animation

```jsx
import { useScroll, useTransform, useSpring, motion } from "motion/react";

const ref = useRef(null);
const { scrollYProgress } = useScroll({ target: ref, offset: ["start end", "end start"] });
const y = useTransform(scrollYProgress, [0, 1], [0, -300]);     // parallax
const smooth = useSpring(scrollYProgress, { stiffness: 100, damping: 30 });
return <motion.div ref={ref} style={{ y, scaleX: smooth }} />;
```

In recent versions, `useScroll`/`scroll()` can run on the browser's **ScrollTimeline API** for hardware-accelerated, off-main-thread scroll animation — pass `scrollYProgress` directly to `opacity` or through `useTransform` to a transform/`filter`. Recent versions also animate modern color types (oklch/oklab/lab/lch/color-mix).

## Performance and accessibility

- Animate transform/opacity; avoid animating layout props except via the `layout` prop (which uses transforms under the hood).
- Shrink the bundle with **`LazyMotion`** + the **`m`** component (load features lazily) — meaningfully smaller than the full `motion` import.
- Virtualize lists beyond ~50 animated items.
- Respect motion preferences: wrap the app in `<MotionConfig reducedMotion="user">` or branch on `useReducedMotion()`. Use `whileFocus` so keyboard users get equivalent affordances.
- Known build quirk: on some edge runtimes (e.g. Cloudflare Workers) you may need to pin a specific `framer-motion`/`motion` patch version if a build breaks.
