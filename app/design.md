# Examiner's Desk: design system

## Concept
**An examiner's marking desk.** The student's answer booklet lies open on the left; the
examiner's marking sheet sits beside it. Student writing is shown as it was written (crops of
the real page). Everything the system concludes is written in the margin the way an examiner
would: ticks for matched concepts, a half tick for partial, a red ring for missing, a total
boxed at the top of the page. Gold is the examiner's seal: it marks the final award and the
one primary action per screen, nothing else.

Different from the previous projects on this machine: no navy instrument readouts
(Chromatonic), no plum console with dot texture (Upgence). Warm paper, no texture, no dots.

## Deliberate choices (anti-slop register)
| Choice | Reason |
|---|---|
| Gold accent on warm white | Requested by the product owner (2026-10-28). Kept to the seal role above so the page never becomes "the golden template": data colours carry meaning, gold never does. |
| Inter for all text | Requested by the product owner. The screen is mostly figures (marks, scores, confidences); Inter's tabular numerals (`font-variant-numeric: tabular-nums`) keep columns aligned. Self-hosted, latin subset, OFL. |
| No icons library | The few glyphs needed (tick, half tick, cross) are typographic characters set in the ink colours. |

## Colour tokens (defined once in `static/styles.css` `:root`)
| Token | Hex | Meaning |
|---|---|---|
| `--paper` | `#FAF7F0` | page background (never pure white) |
| `--sheet` | `#FFFDF8` | panels, the marking sheet |
| `--sheet-sunk` | `#F3EEE2` | wells: crops, inputs at rest, table heads |
| `--rule` | `#E4DCCB` | hairline borders |
| `--rule-strong` | `#CFC3AA` | dividers between major regions |
| `--ink` | `#1E1B16` | body text |
| `--ink-2` | `#5E574B` | secondary text, labels |
| `--ink-3` | `#8C8474` | captions, disabled |
| `--gold` | `#A87B1F` | the seal: final marks, primary button |
| `--gold-deep` | `#7E5A10` | gold text on paper, pressed state |
| `--gold-wash` | `#F4E8C8` | seal background, focus ring halo |
| `--pen-blue` | `#23408E` | student text as transcribed (the student's pen) |
| `--ok` | `#2F6B3A` | matched concept, positive attribution |
| `--ok-wash` | `#E2EFDF` | |
| `--part` | `#9A5B12` | partially matched, low-confidence word |
| `--part-wash` | `#F7E6CC` | |
| `--miss` | `#A3302A` | missing concept, negative attribution, review flag |
| `--miss-wash` | `#F6DEDA` | |

Data colours (`ok/part/miss`) are only used for data meaning. No hex literals outside `:root`.

## Type scale (Inter)
| Role | Size / line | Weight | Notes |
|---|---|---|---|
| Award figure | 44 / 48 | 600 | tabular, gold-deep |
| Page title | 20 / 28 | 600 | letter-spacing -0.01em |
| Section head | 12 / 16 | 600 | uppercase, tracking 0.08em, ink-2 |
| Body | 14 / 22 | 400 | |
| Dense / table | 13 / 20 | 400 | tabular numbers |
| Caption | 12 / 16 | 400 | ink-3 |

## Spacing, radius, borders
- Spacing scale (px): 4, 8, 12, 16, 24, 32, 48. Panels pad 24; table cells 8 x 12.
- Radius per component: panels 2 (paper sheets are square), inputs 4, buttons 4, chips 999 only
  for status chips, crops 0.
- Borders are 1px `--rule`. Shadows: none, except the seal (`0 1px 0 var(--rule-strong)`).

## Components
- **Seal**: boxed final award, gold border 1.5px, `marks / total`, underneath the breakdown line.
- **Button**: primary (gold fill, sheet text), secondary (sheet fill, rule border, ink text),
  quiet (text only, ink-2, underline on hover). One primary per screen.
- **Status chip**: `ok / part / miss` wash background with matching ink, 12px, 600.
- **Line row**: crop on the left at natural aspect ratio, editable transcription on the right in
  `--pen-blue`; low-confidence words get a `--part` dotted underline; alternative engine
  readings listed under it in ink-3 with the engine name.
- **Margin note**: review reasons, set in `--miss` with a thin left rule (not a coloured card).
- **Skeleton**: sunk blocks in the exact shape of the results layout, while evaluating.
- **Empty state**: a short line of instruction where results will appear.

## Motion
One motion: when an award is confirmed or overridden, the seal settles (scale 1.04 -> 1,
opacity 0.6 -> 1, 180 ms). Off under `prefers-reduced-motion`.

## Copy
Plain, concrete examiner language: "Marks awarded", "Needs your review", "Key points".
No em dashes in UI copy.
