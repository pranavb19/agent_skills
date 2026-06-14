# Craft: Accessibility

Accessibility is polish, not paperwork. The same care that makes a UI feel crafted — clear focus, logical order, respect for user intent — is what makes it usable for everyone.

## Table of contents
- Semantic HTML first
- Visible focus states
- Keyboard navigation
- Reduced motion
- Color and contrast
- Respecting user intent
- Quick checklist

## Semantic HTML first

Use real elements: `<button>` for actions (not a `<div onClick>`), `<a>` for navigation, `<nav>`, `<main>`, `<header>`, `<footer>`, headings in correct order (one `<h1>`, no skipped levels). Semantic HTML gives you keyboard operability, roles, and screen-reader semantics for free. Reach for ARIA only to fill genuine gaps — incorrect ARIA is worse than none. Primitive libraries (Radix, used by shadcn/ui) supply correct roles/states/focus management out of the box; prefer them for menus, dialogs, comboboxes, tabs.

## Visible focus states

Never remove an outline without replacing it. Design a deliberate, on-brand focus ring using `:focus-visible` (so it shows for keyboard users without cluttering mouse clicks):

```css
:focus-visible {
  outline: 2px solid var(--primary);
  outline-offset: 2px;
  border-radius: 4px;
}
:focus:not(:focus-visible) { outline: none; }
```

The ring must have 3:1 contrast against adjacent colors. A distinctive focus style is a craft detail, not an eyesore.

## Keyboard navigation

Everything interactive must be operable without a mouse, in a logical tab order (follow DOM order; avoid positive `tabindex`). Manage focus deliberately:
- Move focus into modals/dialogs/menus on open and trap it while open; restore focus to the trigger on close (Radix/shadcn handle this).
- Provide a visible "skip to content" link as the first focusable element.
- Custom widgets need the expected key bindings (Esc to close, arrows to move within a menu, Enter/Space to activate).

## Reduced motion

See `craft-motion.md`. Summary: honor `prefers-reduced-motion: reduce` everywhere; prefer the opt-in pattern; replace movement with crossfades; keep functional feedback (loading, error, success) intact even when motion is off; never trigger big zoom/spin/parallax for reduced-motion users.

## Color and contrast

WCAG AA: 4.5:1 for normal text, 3:1 for large text (≥24px or ≥18.66px bold) and for UI components/icons/borders that convey meaning. Don't rely on color alone — pair it with text, icon, or shape (e.g., error states get an icon and a message, not just red). Verify final pairings, especially muted text on tinted/elevated surfaces and accent-on-surface for CTAs.

## Respecting user intent

- Don't trap or hijack scroll; if you use smooth scroll (Lenis), keep keyboard scrolling, anchor jumps, and `position: sticky` working.
- Don't autoplay disorienting motion or media with sound.
- Keep looping animations pausable; provide controls for carousels/video.
- Respect `prefers-color-scheme` for dark mode default.
- Forms: real `<label>`s (associated via `for`/`id`), `aria-describedby` for hints/errors, `aria-invalid` on bad fields, errors announced (`role="alert"` or live region), inputs with appropriate `type`/`autocomplete`/`inputmode`.

## Quick checklist

- [ ] Semantic elements; headings in order; one `<h1>`.
- [ ] Every control reachable and operable by keyboard, logical tab order.
- [ ] Visible `:focus-visible` ring with sufficient contrast.
- [ ] Focus managed in overlays; skip link present.
- [ ] AA contrast on text, UI, and focus ring; meaning never color-only.
- [ ] `prefers-reduced-motion` honored; functional feedback preserved.
- [ ] Images have meaningful `alt` (empty `alt=""` for decorative).
- [ ] Tested with keyboard only and with a screen reader on key flows.
