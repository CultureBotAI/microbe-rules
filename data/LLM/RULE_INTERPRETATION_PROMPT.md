# Rule-interpretation prompt (canonical)

Use this for interpreting mined association rules for a predicted taxon→medium
**combination** — one organism paired with one medium, which is the unit the
rule mining produces (seven of them in the current set). It replaces the four
per-medium files in this folder
(`LLM_514_Table_8`, `LLM_514_Table_11`, `LLM_65_Table_7`, `LLM_65_Table_10`),
which are kept for reference because the published tables were produced with
them.

It is **medium-agnostic**: fill in section F and it works for any of the seven
combinations. It asks for the two things the earlier prompts did not:
relationships *between* features, and a categorisation of the case rather than
a one-off reading.

**Do not edit sections B–E per medium.** Only section F changes. The earlier
prompts were copied per medium and drifted — one of them still tells the model
to label a medium-65 result as DSMZ 514.

## How to run it

1. Fill in every `<...>` placeholder in **F. Data**. Leave a heading out
   entirely if you have nothing for it; do not leave a placeholder in place.
2. Paste the whole file, F included, as a single message.
3. Run it **once per combination** — one organism with one medium. Do not put
   two combinations in the same request: the aggregation in B1 is over the rule
   set of a single combination, and combining them silently averages two
   different organisms into one answer.
4. Keep the model's reply as-is. The rule statistics in it are yours to check;
   the biology is for the CultureBot side to check.

Model: any current frontier model. We use Claude Opus 5 and it handles this
well. Note its safety filter sometimes stops generation on rule sets dense with
clinical isolation terms (wound, lung, inflammation); Claude Opus 4.8 clears
that and stays comparable. A 13b local model will not do this job well.

---

## A. Context

You are given the **complete set** of mined association rules that fire for one
predicted **combination** of one microbial taxon with one growth medium (the
unit the rule mining produces). Each rule has the
form `Condition1 & Condition2 ⇒ grows on <medium>`, with support, confidence and
lift. Features are statements about the organism drawn from a knowledge graph:
chemical substrates (`CHEBI:`), enzyme activities (`EC:`), isolation source and
host context, growth physiology (oxygen, temperature, salinity), and environment
(`ENVO:`). The medium's ingredient list is given too.

Association rules are **correlational**. They record that a feature co-occurs
with growth in the training data, not that it causes growth.

## B. Task

Produce four sections, in this order.

**B1. Rationale** — one paragraph, 4–6 sentences, synthesising across the
**whole** rule set rather than the strongest rule. Which categories of feature
dominate, and how do they cohere into a biological account of growth on this
medium?

**B2. Dominant feature categories** — one bullet per category that carries
signal, from: chemical substrates, enzyme activities, host/isolation context,
growth physiology, environment. Within each bullet, **roll members up to their
shared parent class** where one exists, naming the parent and then its members,
rather than listing features flat. Give the share of rules each occupies.

**B3. Relationships between features** — this is the section the earlier prompts
lacked. Surface connections, not a list:
- **Hierarchical**: several features sharing one parent class (e.g. several
  hexoses under *D-aldohexose*; several glycoside hydrolases under *EC:3.2.1*).
- **Enzyme ↔ substrate**: an enzyme activity and a chemical it acts on, where
  **both** appear as features.
- **Environment ↔ physiology**: isolation context co-occurring with oxygen,
  temperature or salinity traits.
- **Feature ↔ medium composition**: a feature corresponding to a named
  ingredient of this medium.
- **Rule structure**: do the rules converge on one shared anchor condition, or
  branch into independent routes? Say which, and name the anchor if there is one.
If a relationship type is absent, say so explicitly rather than omitting it.

**B4. Case categorisation** — classify this combination on three axes, and give one
clause of justification for each.

- *Evidential basis* — choose one:
  - **Composition-matched**: chemical or enzyme features correspond to named
    ingredients of this medium.
  - **Habitat-proxy**: the signal is ecological or isolation-source; nothing
    connects to the medium's composition.
  - **Physiology-gated**: oxygen, temperature or salinity traits carry the
    signal.
  - **Mixed**: more than one of the above contributes materially.
- *Rule-set shape* — **convergent** (rules share an anchor and differ only in a
  qualifier) or **branching** (independent routes to the same conclusion).
- *Confidence standing* — **only if the optional model block in F is present.**
  Relate the model probability to the decision threshold, and to the support and
  confidence of the rules; state whether the prediction was correct, and whether
  the rules explain the outcome or merely accompany it. **If that block is
  absent, write "not assessable — no model output supplied" and do not
  substitute a guess.** The first two axes do not need it and are always
  assessable from the rules alone.

**B5. Critique** — two sentences. What would make this interpretation wrong, and
what is the weakest step in the argument?

## C. Referencing

1. Refer to rules by their IDs, e.g. #1, #2.
2. Use feature names **exactly** as they appear in section F, e.g.
   `ENVO:sea water`, `isolation source: marine`.
3. Name medium ingredients when a feature relates to one.
4. **Do not introduce any `CHEBI:`, `EC:` or `ENVO:` identifier that does not
   appear verbatim in section F.** If you want to name a parent class you have
   not been given an ID for, use its name in words and do not invent a number.
   Fabricated identifiers are the single most common failure in this task.
5. Do not assert mechanism. Write "co-occurs with", "is consistent with",
   "would be expected to" — not "causes" or "enables".

## D. Output format

```
Medium: <medium name and DSMZ number, copied from section F>
Organism: <organism name and taxon ID, copied from section F>

## Rationale
<B1>

## Dominant feature categories
- <category>: <parent class, then members> — <what it implies>

## Relationships between features
- Hierarchical: <...>   (or: none present)
- Enzyme ↔ substrate: <...>   (or: none present)
- Environment ↔ physiology: <...>   (or: none present)
- Feature ↔ medium composition: <...>   (or: none present)
- Rule structure: <convergent | branching>, <anchor if any>

## Case categorisation
- Evidential basis: <one of the four> — <justification>
- Rule-set shape: <convergent | branching> — <justification>
- Confidence standing: <...>

## Critique
<B5>
```

## E. Summary table

End with a table with exactly these columns:

| feature or feature group | members | shared property | related features | association with medium |

One row per feature group rather than per feature. In **related features**, name
the features this group is connected to and the relationship type. In
**association with medium**, put any medium component in **bold**.

## F. Data

Fill in what you have. **Delete any block you cannot fill — never leave a
`<...>` placeholder in place**, because the model will fill it in for you.

The medium recipes are already in the four per-medium prompts in this folder;
copy the ingredient table from the one for your medium.

```
Organism: <name> (<NCBITaxon:...>)
Predicted medium: <name> (DSMZ <id>)

--- OPTIONAL: model outputs. Delete these two lines if you do not have them.
--- They come from the prediction side, not from the rule mining. Without them
--- the confidence axis in B4 is reported as not assessable, which is correct.
Model probability: <...>          Decision threshold: <...>
Observed growth: <...>            Outcome vs ground truth: <TP | FN | FP | TN>
--- END OPTIONAL

Medium ingredients (compound, amount, unit, g/L, mM):
<paste the recipe table from the per-medium prompt in this folder>

FULL RULE SET — <N> rules fire for this combination. Interpret the aggregate.
<paste the rules, with IDs, support, confidence, lift>

Feature frequency across the full rule set
(feature; #rules containing it; % of rules; max confidence; max lift):
<paste>

Most frequent co-occurring feature pairs (pair; #rules):
<paste>
```
