# Craft: Avoiding AI Slop

The catalogue of generic tells and the concrete fix for each. The root cause of slop is statistical: reaching for the most common pattern in the training data (the average of every Tailwind tutorial 2019–2024). The cure is **constraint and intentionality** — fewer, more opinionated decisions, executed fully.

## Table of contents
- The tells and their fixes
- The three antidotes: restraint, hierarchy, commitment
- The self-audit
- Worked before/after examples

## The tells and their fixes

| Slop tell | Why it reads as generic | Fix |
|---|---|---|
| Inter/Roboto/Arial everywhere, one weight | No voice, no hierarchy | Distinctive display face + refined body face; a real modular scale; deliberate weight contrast |
| Purple→blue gradient on white | The single most overused AI palette | Commit to one dominant color; OKLCH palette; sharp sparing accent |
| Gradient text on headings/numbers | Decoration substituting for hierarchy | Solid color; create emphasis with scale/weight/space |
| Centered everything + small pill badge | Default safe layout | Asymmetry; left-aligned editorial layouts; one anchor per section |
| Three identical feature cards | Template grid | Vary card sizes (bento), break the grid, differentiate content density |
| Uniform shadow + uniform 16px radius on all surfaces | Page goes flat; nothing reads above anything | Real elevation system; vary radius by component role |
| Emoji in headings (🚀 ✨ 🔥) | Tutorial aesthetic | Real icons (Lucide) or none; let type carry tone |
| Indiscriminate glassmorphism blur | Applied without purpose | Use blur only where depth is real (overlays, sticky bars) |
| `transition: all` on everything | Mushy, janky, slow | Animate only `transform`/`opacity`, named properties, tuned easing |
| Raw default Tailwind palette | Looks like every starter | Define semantic tokens from a custom OKLCH base |
| Evenly distributed timid palette | No focal point | 60/30/10: dominant / secondary / sharp accent |
| Generic stock-y hero illustration | Placeholder energy | Real product UI, a single bold type treatment, or one crafted visual |

## The three antidotes

**Restraint.** One memorable visual anchor per section beats scattered micro-interactions. One well-orchestrated page-load reveal (staggered, transform/opacity) creates more delight than ten hover wiggles. If you can remove an effect and the page is still clear, remove it.

**Hierarchy.** Slop is flat — equal shadows, equal sizes, equal spacing, so the brain can't triage. Craft uses elevation, scale, contrast, and space to make importance legible *without reading*. Every screen should have an obvious first thing, second thing, third thing.

**Commitment.** Dominant colors with sharp accents outperform timid, evenly-distributed palettes. Pick a real aesthetic — editorial, brutalist, soft-minimal, technical/mono, maximalist — and execute it fully rather than hedging toward the safe middle.

## The self-audit (run before finishing any UI)

- Did I replace the default font and the default palette, or did I ship starter defaults?
- Is there a clear visual hierarchy, or is the page flat (uniform shadows/sizes)?
- Is there one deliberate, opinionated decision per section, or is everything centered and symmetric?
- Did I commit to a dominant color, or hedge with an even rainbow?
- Is motion purposeful (orientation/feedback/continuity) or decorative? Does it respect `prefers-reduced-motion`?
- Would a designer recognize an intentional direction here, or would they call it "AI-generated"?

If any answer points to the default, introduce one opinionated decision and execute it cleanly.

## Worked example

**Before (slop):** centered hero, Inter 700 headline, purple→blue gradient text, pill badge "🚀 Now in beta", three cards with `shadow-md rounded-2xl`, CTA `bg-gradient-to-r from-purple-500 to-blue-500`.

**After (craft):** left-aligned hero on an off-black `oklch(0.18 0.01 260)` surface; oversized display headline in a distinctive grotesk with tight negative tracking; one accent color `oklch(0.72 0.19 35)` used only on the primary CTA and a single underline; bento layout (one large tile + two small) with differentiated elevation; CTA is solid accent with a 160ms `transform: translateY` + shadow lift on hover, disabled under reduced motion. Same content, opposite impression.

## Choosing a direction

Before building, pick ONE aesthetic lane and commit:
- **Editorial** — strong type, generous measure, asymmetric grid, restrained color.
- **Technical/mono** — monospace accents, hairline borders, data density, minimal motion.
- **Soft-minimal** — large radius, soft shadows, muted OKLCH pastels, gentle springs.
- **Brutalist** — raw type, hard edges, high contrast, deliberate "unfinished" feel.
- **Maximalist/immersive** — 3D/WebGL hero, scroll storytelling, bold motion (budget carefully).

The lane dictates type, color, spacing, and motion together — that coherence is what defeats slop.
