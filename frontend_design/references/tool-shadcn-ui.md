# Tool: shadcn/ui

The design-system foundation: a CLI that copies accessible component source into your repo (you own the code), built on Radix (or Base UI) primitives + Tailwind. This own-the-code model won because it gives accessibility and a consistent token system without vendor lock-in.

## Table of contents
- The model
- Setup
- Theming with tokens
- 2025–2026 developments (registries)
- Why it's a good foundation
- Composing with motion and effect libraries

## The model

shadcn/ui is **not an npm dependency**. You run a CLI that drops component source into `components/ui/`, so you can edit anything. Components are built on **Radix UI / Base UI primitives** (accessible behavior: focus management, keyboard, ARIA) styled with **Tailwind**. You get accessibility for free and full control over markup and styling.

## Setup

```bash
npx shadcn@latest init          # pick base color, CSS variables, Radix or Base UI
npx shadcn@latest add button dialog dropdown-menu
```
`init` writes a `components.json` (style, base color, token strategy, paths) and sets up your CSS variables and `cn()` utility.

## Theming with tokens

Theming uses semantic CSS variables so dark mode is a variable swap, not a component rewrite. With OKLCH for predictable lightness:

```css
:root {
  --background: oklch(0.99 0.005 260);
  --foreground: oklch(0.20 0.02 260);
  --primary:    oklch(0.55 0.20 264);
  --primary-foreground: oklch(0.98 0 0);
  --muted:      oklch(0.96 0.008 260);
  --border:     oklch(0.90 0.01 260);
  --radius: 0.625rem;
}
.dark {
  --background: oklch(0.18 0.01 260);
  --foreground: oklch(0.95 0.01 260);
  --primary:    oklch(0.72 0.19 264);   /* brighter on dark */
}
```
Components reference `bg-background`, `text-foreground`, `bg-primary` — change the variables and the whole system retheme. This is the cleanest way to escape the "default Tailwind palette" slop look: define your own tokens here (see `craft-color.md`).

## 2025–2026 developments (registries)

- **Tailwind v4** support.
- Choice of **Radix or Base UI** primitives at init; unified `radix-ui` package.
- **Namespaced registries** — pull components from multiple sources, including Aceternity and Magic UI:
  ```json
  // components.json
  "registries": {
    "@aceternity": "https://ui.aceternity.com/registry/{name}.json",
    "@magicui": "https://magicui.design/r/{name}.json"
  }
  ```
  then `npx shadcn@latest add @magicui/marquee`.
- **`shadcn build`** to publish your own registry; an **MCP server** for AI assistants; RTL support; OKLCH-based theming.

## Why it's a good foundation

- Accessibility from Radix/Base UI primitives (the hard part, done right).
- You own the code — no fighting a library's opinions; edit freely.
- Token-driven theming — consistent, easy dark mode, easy brand application.
- Composes with the whole ecosystem (Aceternity, Magic UI) via the shared registry/token model.

## Composing with motion and effect libraries

shadcn handles structure + a11y; Motion handles movement. Wrap or compose primitives with `motion.*`, use `AnimatePresence` for dialogs/sheets (keep it mounted, unique keys), and don't unmount focus-trapped content mid-animation (preserve Radix focus management). For "wow" effects, pull selectively from Aceternity/Magic UI and **restyle to your tokens** so the result isn't the generic template look. See `tool-component-libraries.md` and `combining-tools.md`.
