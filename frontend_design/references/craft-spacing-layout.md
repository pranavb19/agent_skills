# Craft: Spacing & Layout

Spacing is invisible when right and glaring when wrong. Random pixel values and flat, symmetric grids are slop; a consistent scale and deliberate hierarchy are craft.

## Table of contents
- The spacing scale (8-point grid)
- Whitespace as a feature
- Optical alignment
- Visual hierarchy and elevation
- Breaking symmetry
- Layout patterns

## The spacing scale (8-point grid)

Use a fixed spacing scale, not arbitrary values. The 8-point grid is the common standard: 4 / 8 / 16 / 24 / 32 / 48 / 64 / 96 / 128. Everything (padding, gaps, margins) snaps to these steps, which makes the whole layout feel coherent.

```css
:root {
  --space-1: 0.25rem;  --space-2: 0.5rem;  --space-3: 1rem;
  --space-4: 1.5rem;   --space-5: 2rem;    --space-6: 3rem;
  --space-7: 4rem;     --space-8: 6rem;    --space-9: 8rem;
}
```

Consider fluid spacing with `clamp()` for section padding so vertical rhythm scales with the viewport:
```css
.section { padding-block: clamp(3rem, 5vw + 1rem, 8rem); }
```

## Whitespace as a feature

Generous, *intentional* negative space reads as premium; cramped layouts read as cheap. Whitespace is not wasted — it groups related items (proximity), separates unrelated ones, and gives the eye somewhere to rest. When something feels off, the answer is usually more space around the important element, not more decoration.

## Optical alignment

Mathematical alignment is not always *visual* alignment. Nudge icons, arrows, quotation marks, and circular/triangular shapes so they *look* centered even when the bounding boxes aren't. Optical adjustments (a few px) are a hallmark of hand-crafted work.

## Visual hierarchy and elevation

Establish hierarchy through scale, weight, color, and space — and through a real **elevation system**. Slop uses one shadow everywhere, so the page is flat. Define elevation tiers:

```css
:root {
  --elev-0: none;
  --elev-1: 0 1px 2px oklch(0 0 0 / 0.06), 0 1px 1px oklch(0 0 0 / 0.04);
  --elev-2: 0 4px 12px oklch(0 0 0 / 0.08), 0 2px 4px oklch(0 0 0 / 0.04);
  --elev-3: 0 12px 32px oklch(0 0 0 / 0.12), 0 4px 8px oklch(0 0 0 / 0.06);
}
```
Higher elevation = more important / more foreground (modals, popovers). Use sparingly; soft, layered shadows beat one hard drop shadow. In dark mode, prefer lighter surfaces over heavy shadows to signal elevation.

## Breaking symmetry

Centered-everything is the safe default and reads as generic. Introduce intentional asymmetry: left-aligned editorial hero, off-center focal points, a bento grid (one large tile + smaller ones) instead of three identical cards, content that breaks the grid at one deliberate point. One memorable anchor per section.

## Layout patterns

- **CSS Grid for 2D structure**, Flexbox for 1D flows. Use `grid-template-columns: repeat(12, 1fr)` as a base and span deliberately.
- **Container queries** (`@container`) for components that must adapt to their slot, not just the viewport.
- **`gap`** over margins for spacing between flex/grid children (no margin-collapse surprises).
- **`min()`/`max()`/`clamp()`** for fluid, breakpoint-light layouts: `width: min(100% - 2rem, 70ch)`.
- Reserve space for async/media content (`aspect-ratio`) so layout never shifts (CLS).
- Define a consistent content measure and gutter, then let sections vary *within* that frame.
