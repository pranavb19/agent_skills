# Tool: Component Libraries (Aceternity UI & Magic UI)

Copy-paste animated components for landing-page "wow" effects, built on Tailwind + Motion and installed through the shadcn CLI. Powerful accelerators — but a real source of "every site looks the same" if used verbatim.

## Table of contents
- How they're structured
- Aceternity UI
- Magic UI
- The craft tension (read this)
- Integration and performance

## How they're structured

Both distribute through the **shadcn CLI registry** model: you run a command, the component source is copied into your repo, and you own/edit it. They're built with **React + TypeScript + Tailwind + Motion (Framer Motion)**. Add the registry to `components.json`, then `add` components by namespace.

## Aceternity UI

200+ animated components and templates. Install via namespaced registry:
```json
"registries": { "@aceternity": "https://ui.aceternity.com/registry/{name}.json" }
```
```bash
npx shadcn@latest add @aceternity/background-beams
```
Notable: 3D Card Effect, Hero Parallax, Macbook Scroll, Lamp Effect, Background Beams, Aurora Background, Sparkles, Tracing Beam, Infinite Moving Cards, Bento Grid, Moving Border, Text Generate Effect, Spotlight. All are **client components** (`"use client"`, they use Motion).

## Magic UI

150+ free, open-source (MIT) animated components built with React, TypeScript, Tailwind, and Motion — pitched as a companion to shadcn/ui. Same install model:
```bash
npx shadcn@latest add @magicui/marquee   # or @magicui/globe, @magicui/dock, ...
```
Notable: Marquee, Bento Grid, Animated Beam, Border Beam, Shimmer Button, Particles, Globe, Dock, Orbiting Circles, Animated List, Terminal, Number Ticker, Meteors, Retro Grid, device mocks (Safari/iPhone/Android), and text animations (Aurora Text, Sparkles Text, Morphing Text). A paid **Magic UI Pro** offers prebuilt templates/landing sections under a one-time commercial license.

## The craft tension (read this)

These libraries make "everyone's site look the same." Background Beams + Aurora background + a Bento grid + a typewriter headline is now itself an instantly recognizable cliché — a more elaborate flavor of slop. Use them as **accelerators, not identity**:
- **Restyle to your tokens** — change colors, type, radius, spacing so it matches your design system (see `craft-color.md`, `tool-shadcn-ui.md`).
- **Pick ONE signature effect** per page and keep the rest of the design restrained.
- **Delete** the components you don't use; don't ship a kitchen sink.
- Consider hand-building the hero interaction with Motion/GSAP when you want genuine differentiation — a library effect with default colors reads as generic.

## Integration and performance

- All are client components — they won't SSR. **Lazy-load heavy animated backgrounds** (`next/dynamic`, `ssr: false`) so they don't block first paint, and gate them on viewport.
- They depend on Motion; if you also use Motion elsewhere, you share the runtime.
- Respect `prefers-reduced-motion` — many of these effects (beams, meteors, particles) are decorative and should be disabled or simplified for reduced-motion users (see `craft-motion.md`).
- Budget them like any animation library against LCP/INP (see `craft-performance.md`).
