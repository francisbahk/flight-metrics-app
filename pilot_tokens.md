# Pilot Study Tokens

## Overview
- **3 groups** of 5 participants each (15 total tokens)
- **Group A**: Complete flight search and ranking (no re-ranking)
- **Group B**: Complete flight search + re-rank 4 prompts from Group A
- **Group C**: Complete flight search + re-rank 4 prompts from Group B

## Balanced Coverage
Each prompt from the previous group is reviewed by exactly 4 participants.

---

## Group A Tokens (No Re-ranking)

| Token | Re-rank Targets |
|-------|-----------------|
| `GA01` | None (first group) |
| `GA02` | None (first group) |
| `GA03` | None (first group) |
| `GA04` | None (first group) |
| `GA05` | None (first group) |

---

## Group B Tokens (Re-rank Group A)

| Token | Re-rank Targets | Skips |
|-------|-----------------|-------|
| `GB01` | GA02, GA03, GA04, GA05 | GA01 |
| `GB02` | GA01, GA03, GA04, GA05 | GA02 |
| `GB03` | GA01, GA02, GA04, GA05 | GA03 |
| `GB04` | GA01, GA02, GA03, GA05 | GA04 |
| `GB05` | GA01, GA02, GA03, GA04 | GA05 |

---

## Group C Tokens (Re-rank Group B)

| Token | Re-rank Targets | Skips |
|-------|-----------------|-------|
| `GC01` | GB02, GB03, GB04, GB05 | GB01 |
| `GC02` | GB01, GB03, GB04, GB05 | GB02 |
| `GC03` | GB01, GB02, GB04, GB05 | GB03 |
| `GC04` | GB01, GB02, GB03, GB05 | GB04 |
| `GC05` | GB01, GB02, GB03, GB04 | GB05 |

---

## Coverage Matrix

### Group A prompts reviewed by Group B:

| Prompt | Reviewed By |
|--------|-------------|
| GA01 | GB02, GB03, GB04, GB05 |
| GA02 | GB01, GB03, GB04, GB05 |
| GA03 | GB01, GB02, GB04, GB05 |
| GA04 | GB01, GB02, GB03, GB05 |
| GA05 | GB01, GB02, GB03, GB04 |

### Group B prompts reviewed by Group C:

| Prompt | Reviewed By |
|--------|-------------|
| GB01 | GC02, GC03, GC04, GC05 |
| GB02 | GC01, GC03, GC04, GC05 |
| GB03 | GC01, GC02, GC04, GC05 |
| GB04 | GC01, GC02, GC03, GC05 |
| GB05 | GC01, GC02, GC03, GC04 |

---

## Usage Instructions

1. **Group A goes first**: Distribute GA01-GA05 tokens to first 5 participants
2. **Wait for Group A to complete**: All 5 must finish before Group B starts
3. **Group B goes second**: Distribute GB01-GB05 tokens to next 5 participants
4. **Wait for Group B to complete**: All 5 must finish before Group C starts
5. **Group C goes last**: Distribute GC01-GC05 tokens to final 5 participants
