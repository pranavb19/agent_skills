# Craft: Typography

Typography is the fastest way to look intentional — or generic. Default fonts everywhere is the single biggest slop tell.

## Table of contents
- Escaping the defaults
- Type scale (modular)
- Vertical rhythm and line-height
- Fluid typography with clamp()
- Measure, tracking, and variable fonts
- Practical defaults

## Escaping the defaults

Inter/Roboto/Arial-for-everything has no voice. Choose a **distinctive display face** for headings paired with a **refined, readable body face**. Even system fonts can feel deliberate if hierarchy, tracking, and scale are handled with care. Pairings that read as designed: a characterful grotesk or serif display + a neutral, legible body; a mono accent for labels/code. Limit to two families (plus an optional mono).

## Type scale (modular)

Use a modular scale (a fixed ratio) rather than arbitrary px values. Common ratios: 1.2 (minor third), 1.25 (major third), 1.333 (perfect fourth — more dramatic). Define as CSS variables in `rem`.

```css
:root {
  --ratio: 1.25;
  --fs-0: 1rem;                               /* body */
  --fs-1: calc(var(--fs-0) * var(--ratio));   /* ~1.25rem */
  --fs-2: calc(var(--fs-1) * var(--ratio));   /* ~1.563rem */
  --fs-3: calc(var(--fs-2) * var(--ratio));   /* ~1.953rem */
  --fs-4: calc(var(--fs-3) * var(--ratio));   /* ~2.441rem */
}
```

## Vertical rhythm and line-height

- Body line-height ~1.4–1.6; align spacing to a consistent baseline (e.g. multiples of a 24px line).
- Tighten heading line-height (~1.05–1.2) — big type needs less leading.
- Keep the rhythm consistent: paragraph spacing, list spacing, and section spacing should be multiples of the same base unit.

## Fluid typography with clamp()

`clamp(MIN, PREFERRED, MAX)` scales type/space smoothly across viewports and **eliminates the CLS** that media-query font jumps cause. Use `rem` in the ideal value so browser zoom still works for accessibility.

```css
:root {
  --fs-base: clamp(1rem, 0.89rem + 0.31vw, 1.25rem);
  --fs-h1:   clamp(2rem, 1.6rem + 2vw, 3.5rem);
  --fs-h2:   clamp(1.5rem, 1.3rem + 1vw, 2.25rem);
}
h1 { font-size: var(--fs-h1); line-height: 1.05; letter-spacing: -0.02em; }
```

For harmonious steps across the whole range, consider a Utopia-style approach: define *two* scales (a gentler ratio on phones, a larger ratio on desktop) so the steps interpolate cleanly rather than one viewport looking cramped. Always test zoom up and down.

## Measure, tracking, and variable fonts

- **Measure (line length):** aim for ~60–75 characters for body text. Constrain with `max-width: 65ch`.
- **Letter-spacing as signal:** tight negative tracking (`-0.02em` to `-0.04em`) on large display headings reads as confident; slightly open tracking (`0.05em–0.1em`) on small uppercase labels reads as refined. Body text: leave it alone.
- **Variable fonts:** let you fine-tune weight and optical size cheaply (one file, many weights). Use `font-optical-sizing: auto` and animate `font-variation-settings` sparingly for effects.

## Practical defaults

```css
body {
  font-family: var(--font-body), system-ui, sans-serif;
  font-size: var(--fs-base);
  line-height: 1.6;
  text-rendering: optimizeLegibility;
  -webkit-font-smoothing: antialiased;
}
h1, h2, h3 { font-family: var(--font-display); line-height: 1.1; text-wrap: balance; }
p { text-wrap: pretty; max-width: 65ch; }
```

- `text-wrap: balance` on headings prevents orphans; `text-wrap: pretty` improves paragraph rag.
- Load fonts with `font-display: swap` (or `optional` for non-critical) and preload the primary face to protect LCP.
- Subset fonts to the characters you use to cut bytes.
