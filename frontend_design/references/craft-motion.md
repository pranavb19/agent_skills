# Craft: Motion Design

Motion is the difference between a site that feels alive and one that feels cheap — or nauseating. The craft is timing, easing, and restraint, not effect count.

## Table of contents
- Easing communicates physics
- Duration guidance
- Staggering and choreography
- The 12 principles applied to UI
- When NOT to animate (the frequency gate)
- prefers-reduced-motion (mandatory)
- Motion tokens

## Easing communicates physics

Linear motion feels robotic and wrong for UI. Match the curve to the action:
- **Entering** elements: **ease-out** (arrive fast, settle gently) — `cubic-bezier(0.16, 1, 0.3, 1)` is a great expressive ease-out.
- **Exiting** elements: **ease-in** (start slow, accelerate away).
- **Moving between** states: **ease-in-out**.
- **Springs** feel the most natural and handle interruption/retargeting gracefully — prefer them for interactive, gesture-driven motion (drag, layout).

## Duration guidance

- Routine UI transitions: ~160–240ms.
- Entrances/exits: ~240–360ms.
- Cap most UI motion under ~300ms; longer feels sluggish.
- **Scale duration to distance/area:** a small toggle is faster than a full-screen sheet. Large traversals get longer durations.
- **Exits are shorter than entrances** — get out of the way quickly.

## Staggering and choreography

Stagger grouped reveals by ~50–100ms to create rhythm and direct the eye, rather than everything appearing at once. One well-orchestrated page-load sequence (e.g., headline → subhead → CTA → media, staggered, transform/opacity) creates more delight than scattered hover micro-interactions. Choreograph: things that are related should move together; things that are sequential should cascade.

## The 12 principles applied to UI

- **Timing/spacing (ease):** the foundation — covered above.
- **Anticipation:** a control dips slightly before it acts (`whileTap` scale-down) so the action feels physical.
- **Follow-through / overlapping action:** elements settle with a tiny overshoot or staggered tail rather than stopping dead.
- **Squash & stretch:** subtle scale on press/drop conveys weight (keep it subtle in UI).
- **Arcs:** motion along a slight curve reads more natural than a straight diagonal.
- **Staging:** use motion to direct attention to one thing at a time; don't animate everything simultaneously.

## When NOT to animate (the frequency gate)

This is the most important discipline and the one slop ignores. Ask:
- **How often will the user trigger this?** Rare/expressive actions (onboarding, hero reveal) can be elaborate. Frequent actions (100s/day — list updates, menu opens) should be near-instant or unanimated.
- **Is it keyboard-initiated?** If yes, generally **don't animate** — keyboard users want speed, and motion delays them.
- **Does it serve orientation, feedback, or continuity** — or is it decoration? Decoration in production UI is usually too much.
- **Does it still feel right on the 10th interaction?** If it gets annoying, cut it.

Keep animations interruptible (CSS transitions are interruptible; long keyframe sequences are not). Never block input behind an animation.

## prefers-reduced-motion (mandatory)

Roughly some users have this set for vestibular/comfort reasons. Always honor it. Prefer the **opt-in** pattern so reduced-motion users never see a flash of motion:

```css
/* No motion by default; add it only when the user allows */
.reveal { opacity: 1; }
@media (prefers-reduced-motion: no-preference) {
  .reveal { opacity: 0; transform: translateY(20px); transition: opacity .4s, transform .4s; }
  .reveal.in { opacity: 1; transform: none; }
}
```

Or collapse durations globally via tokens (below). Replace movement with a crossfade rather than removing all feedback — users still need to know something changed. In JS libraries, use the provided hooks (`useReducedMotion()` in Motion, `<MotionConfig reducedMotion="user">`, or gate GSAP timelines).

## Motion tokens

Tokenize durations and easings so the whole system is consistent and reduced motion is one switch:

```css
:root {
  --ease-out: cubic-bezier(0.16, 1, 0.3, 1);
  --ease-in:  cubic-bezier(0.7, 0, 0.84, 0);
  --dur-fast: 160ms;
  --dur-base: 240ms;
  --dur-slow: 360ms;
}
@media (prefers-reduced-motion: reduce) {
  :root { --dur-fast: 0.01ms; --dur-base: 0.01ms; --dur-slow: 0.01ms; }
}
```

Avoid vestibular triggers: large zooms, full-screen spins, aggressive parallax. If you use parallax or scroll-scrubbed motion, soften it heavily and disable it under reduced motion.
