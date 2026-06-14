# Craft: Color

Color is where commitment shows. Raw default palettes and timid even distributions are slop tells; a committed palette in a modern color space is craft.

## Table of contents
- Why OKLCH
- Building a palette
- Semantic tokens
- The 60/30/10 discipline
- Dark mode done right
- Contrast and accessibility

## Why OKLCH

OKLCH is perceptually uniform: equal changes in L (lightness) look equally bright across all hues, unlike HSL where the same lightness value looks wildly different between yellow and blue. This makes programmatic palettes, predictable dark mode, and accessible contrast far easier, and it unlocks wide-gamut P3 colors. Browser support is ~93%+ (Chrome 111+, Safari 15.4+, Firefox 113+); provide a hex fallback or use `@supports`.

```css
/* oklch(Lightness Chroma Hue) — L: 0–1, C: 0–~0.4, H: 0–360 */
:root {
  --brand:     oklch(0.55 0.20 264);
  --brand-300: oklch(0.78 0.13 264);
  --brand-700: oklch(0.39 0.17 264);
}
.btn { background: #5b6cf0; background: var(--brand); } /* hex fallback first */
```

## Building a palette

Don't ship raw Tailwind defaults — that look is its own slop tell. Generate a scale from a single origin so the whole palette feels related. With relative color syntax you can derive tints/shades by only changing L (and slightly C):

```css
:root {
  --accent: oklch(0.70 0.19 35);
  --accent-100: oklch(from var(--accent) 0.95 calc(c * 0.4) h);
  --accent-500: var(--accent);
  --accent-900: oklch(from var(--accent) 0.40 calc(c * 1.1) h);
}
```

Keep hue roughly constant down a ramp; lower chroma at the very light and very dark ends so steps stay natural. A palette is typically: one brand/accent ramp, one neutral ramp (very low chroma, slightly tinted toward the brand hue, not pure gray), plus semantic success/warning/danger.

## Semantic tokens

Map raw colors to role-based tokens so theming is centralized and dark mode is a variable swap, not a component rewrite.

```css
:root {
  --surface:    oklch(0.99 0.005 260);
  --surface-2:  oklch(0.96 0.008 260);
  --text:       oklch(0.20 0.02 260);
  --text-muted: oklch(0.45 0.02 260);
  --border:     oklch(0.90 0.01 260);
  --primary:    var(--brand);
  --primary-fg: oklch(0.98 0 0);
}
```

Components reference `--surface`, `--text`, `--primary` — never raw values. This is exactly how shadcn/ui-style theming works.

## The 60/30/10 discipline

Commit to a dominant color (≈60%, usually a neutral surface), a secondary (≈30%), and a sharp accent (≈10%) used only for the things that matter most (primary CTA, key highlight). Accent scarcity is what makes it powerful. Timid, evenly distributed palettes have no focal point and read as generic.

## Dark mode done right

- **Not pure black.** Use a very dark, slightly chromatic surface (`oklch(0.18 0.01 260)`), with elevation steps where *lighter = higher*.
- **Bump primary lightness** ~15–20% on dark backgrounds so it stays vivid.
- Dark backgrounds tolerate *higher* chroma than light ones.
- A theme swap is just changing the custom properties — no per-component `@media`.

```css
.dark {
  --surface:   oklch(0.18 0.01 260);
  --surface-2: oklch(0.22 0.012 260);  /* one elevation step up */
  --text:      oklch(0.95 0.01 260);
  --primary:   oklch(0.72 0.19 264);   /* brighter than light-mode brand */
}
```

## Contrast and accessibility

Meet WCAG AA: 4.5:1 for body text, 3:1 for large text and UI components/borders. OKLCH's predictable lightness lets you change hue while holding contrast roughly constant. Never encode meaning in color alone — pair color with icon/text/shape. Verify final pairings with a contrast checker, especially for muted text on tinted surfaces.
