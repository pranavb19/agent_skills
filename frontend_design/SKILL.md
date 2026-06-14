---
name: frontend_design
description: A craft-first handbook for building distinctive, high-quality, non-generic frontend web experiences. Use this skill WHENEVER the user asks to build, design, style, or improve any web UI, landing page, marketing site, portfolio, hero section, component, or web app frontend — and ALSO whenever the work involves animation, scroll effects, scroll storytelling, parallax, 3D/WebGL scenes, micro-interactions, page transitions, typography, color palettes, design systems, or making a site feel "polished", "premium", "awwwards-level", "not AI-generated", or "less generic". Trigger this even when the user only names a tool (GSAP, ScrollTrigger, Motion/Framer Motion, React Three Fiber, Three.js, Spline, Lenis, shadcn/ui, Aceternity, Magic UI, Tailwind) or just says "make it look good". The goal is always craft and intentionality, never AI-slop defaults.
---

# Frontend Design

A handbook for building frontend that looks deliberately designed rather than machine-generated. It covers two inseparable halves, weighted equally:

1. **Craft** — typography, spacing, color, motion timing, hierarchy, restraint.
2. **Execution** — performance (Core Web Vitals), accessibility, correct tool integration, clean cleanup.

Decoration without correctness is slop; correctness without taste is forgettable. Aim for both on every task.

## How to use this skill

1. Read this SKILL.md fully — it contains the philosophy, the anti-slop rules, the tool-selection table, and universal rules that apply to *every* frontend task.
2. Then read ONLY the reference files relevant to the current task (listed below). Don't load all of them. Each reference is a self-contained deep-dive with idiomatic, current (2025–2026) code.
3. Apply the craft references (`craft-*.md`) on essentially every visual task — they are what separate good work from slop. Apply tool references when that tool is chosen.

## When to read which reference

| The task involves… | Read |
|---|---|
| ANY visual UI work (almost always) | `references/craft-anti-slop.md`, `references/craft-typography.md`, `references/craft-color.md`, `references/craft-spacing-layout.md` |
| Animation, motion timing, micro-interactions, "feel" | `references/craft-motion.md` |
| Accessibility, focus, keyboard, reduced motion | `references/craft-accessibility.md` |
| Performance, Core Web Vitals, bundle/3D budgeting | `references/craft-performance.md` |
| Scroll storytelling, pinning, scrubbed timelines, parallax | `references/tool-gsap-scrolltrigger.md` |
| React UI animation, gestures, layout/shared-element, presence | `references/tool-motion.md` |
| 3D inside React | `references/tool-react-three-fiber.md` |
| 3D without React / max control / embedding into imperative code | `references/tool-threejs.md` |
| No-code / designer-built 3D, fast interactive hero | `references/tool-spline.md` |
| Silky smooth-scroll feel | `references/tool-lenis.md` |
| Design-system foundation, accessible primitives, theming | `references/tool-shadcn-ui.md` |
| Landing-page "wow" effects, copy-paste animated components | `references/tool-component-libraries.md` |
| Combining several of the above without them fighting each other | `references/combining-tools.md` |

If unsure where to start, read the four always-on craft files plus `references/tool-selection.md`.

## The anti-slop creed (apply to everything)

"AI slop" is the instantly-recognizable median look of ungoverned generation. The cure is not more effects — it is **fewer, more intentional decisions**. The biggest tells to avoid:

- **Default fonts everywhere** (Inter/Roboto/Arial for everything, no hierarchy). Choose a distinctive type pairing and a real type scale.
- **The purple-to-blue gradient on white**, gradient text on numbers, and rainbow accents. Commit to a dominant color with sharp, sparing accents.
- **Centered-everything** hero with a small pill badge, then a tidy three-card feature grid with identical radius and identical drop shadow. Break symmetry; vary elevation; give each section one memorable anchor.
- **Uniform shadows and uniform 16px radius** on every surface — the page goes flat because nothing reads as above anything else. Use a real elevation system.
- **Emoji in headings**, glassmorphism blur applied indiscriminately, and `transition: all` mush.
- **Raw Tailwind default palette** shipped as-is. Define your own tokens.

A reliable instinct: when a layout feels "fine" and generic, you have defaulted. Introduce one deliberate, opinionated decision — an unexpected type treatment, an asymmetric layout, a committed color, a single signature interaction — and execute it cleanly. See `references/craft-anti-slop.md` for the full catalogue and fixes.

## Tool selection (quick reference)

Pick by job; read `references/tool-selection.md` for the expanded table and tradeoffs.

| Use case | Primary tool | Notes |
|---|---|---|
| Scroll storytelling / pin / scrub | **GSAP + ScrollTrigger** | Now fully free incl. all plugins (v3.13, Apr 2025) |
| React UI animation | **Motion** (`motion/react`) | Formerly Framer Motion; springs, layout, presence |
| 3D in React | **React Three Fiber + Drei** | v9 → React 19, v8 → React 18 |
| 3D without React | **Three.js** | r180+, needs import map or bundler |
| No-code 3D | **Spline** | Designer-built; lazy-load, static fallback |
| Silky scroll | **Lenis** | <4kb, wraps native scroll, syncs with GSAP |
| Landing "wow" | **Aceternity / Magic UI** | shadcn-CLI registries; restyle to avoid sameness |
| Design system | **shadcn/ui** | Own-the-code Radix/Base UI + Tailwind |

## Universal rules (every frontend task)

These apply regardless of stack or tool. They are the execution half of craft.

1. **Animate only `transform` and `opacity`** for anything performance-sensitive — they are GPU-composited and don't trigger layout/paint. Never `transition: all`. Reach for `will-change` surgically (it costs memory if overused).
2. **Honor `prefers-reduced-motion: reduce`** on every animation. Prefer the opt-in pattern (no motion by default; add motion only under `prefers-reduced-motion: no-preference`) so reduced-motion users never get a flash. Replace movement with a crossfade rather than killing all feedback. Tokenize durations so reduced motion collapses them globally.
3. **Respect the Core Web Vitals budget**: LCP < 2.5s, INP < 200ms, CLS < 0.1. Heavy animation/3D libraries must be **code-split and lazy-loaded** so they never block the first paint or the first interaction. Reserve space (width/height or `aspect-ratio`) for media to avoid layout shift. See `references/craft-performance.md`.
4. **In React, always clean up.** Use GSAP's `useGSAP()`/`gsap.context().revert()`, kill ScrollTriggers, destroy Lenis, dispose Three.js geometries/materials/textures, give `AnimatePresence` children stable keys. Mark animation/3D components `"use client"` and lazy-load them so SSR/LCP stay fast.
5. **Accessibility is polish, not paperwork.** Semantic HTML first; ARIA only to fill gaps. Always-visible, on-brand `:focus-visible` rings. Everything operable by keyboard. WCAG AA contrast (4.5:1 body, 3:1 large text/UI). Don't trap scroll or autoplay disorienting motion. See `references/craft-accessibility.md`.
6. **Verify versions and licensing at install time** — this ecosystem moves fast. Notably GSAP is now 100% free (all plugins, commercial use) and "Framer Motion" is now "Motion" (`motion` package, `motion/react` import). Don't assume older constraints.
7. **Component libraries are accelerators, not identity.** Aceternity/Magic UI used verbatim with default colors is just more elaborate slop. Restyle tokens, change the type, keep one signature effect, delete the rest.

## Output discipline

- Match the user's stack. Default to React + Next.js (App Router) + Tailwind unless told otherwise; for a standalone-framework tool (vanilla Three.js, Spline runtime) use its native form rather than forcing React.
- Don't dump every effect you know into one page. Choose a coherent direction first (type + color + one signature interaction), then build.
- When you produce a meaningful chunk of UI, sanity-check it against the anti-slop creed and the universal rules before finishing.
- Explain the *why* behind notable design decisions briefly, so the user can carry the taste forward — but keep the focus on the working result.
