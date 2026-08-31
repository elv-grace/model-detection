# 08_embedders — Phase B: which SigLIP 2 checkpoint

Candidates, both within the vectorstore's 1024-d cap (so400m is excluded at 1152-d):

    siglip2-base-patch16-naflex    768-d, aspect-preserving, variable patch count
    siglip2-large-patch16-384     1024-d, fixed 384x384 square

They differ in **two** ways at once — a bigger encoder *and* a different resolution policy — so
neither result cleanly attributes to one of them.

## Verdict: base-naflex. Quality is a tie; it is 6.1x cheaper.

### Same-domain (pool → pool), 1500 brands, 5953-image gallery

| model | dim | r@1 | r@5 | MRR | wide r@1 | GPU ms/crop |
|---|---|---|---|---|---|---|
| base-naflex | 768 | 0.926 | 0.947 | 0.937 | 0.943 | **5.95** |
| large-384 | 1024 | 0.929 | 0.959 | 0.942 | 0.947 | 36.42 |

Binomial SE at n=1500 is 0.013, so anything under ~0.026 is a tie. **r@1 differs by 0.003.**

The cost gap is the decision. At ~17 crops/frame, 5.95 ms/crop is 101 ms of embedding per frame
against 619 ms — the embedder would go from a fifth of the frame budget to more than the
detector costs. Note this is GPU-only on already-decoded crops, which is what the tagger pays.
An end-to-end figure over JPEGs on disk showed 30.7 against 41.2 ms and badly understated the
gap, because decode dominated it.

### Cross-domain (detected crop → pool) — the realistic query

Same-domain retrieval is not what the tagger does. It queries a *detection* — small,
motion-blurred, off-angle — against clean reference art.

| model | dim | r@1 | r@5 | MRR | Gap | KIA | NBA | NFL | Nike |
|---|---|---|---|---|---|---|---|---|---|
| base-naflex | 768 | 0.60 | 0.60 | 0.60 | 1.00 | 0.00 | 0.50 | 0.83 | 0.50 |
| large-384 | 1024 | 0.60 | 0.60 | 0.61 | 1.00 | 0.00 | 0.50 | 0.83 | 0.50 |

**Underpowered by construction: n=15, SE ~0.13.** The footage is NBA/NFL broadcast and its marks
are mostly leagues, teams and US insurers — State Farm, ESPN, NBC, the Chiefs arrowhead,
Buccaneers, USA Basketball — none of which are in the 2960-brand pool. Only NBA, NFL, Nike, KIA
and Gap overlap. This says whether cross-domain retrieval works, not which model is better.

Three things it does show:

1. **Cross-domain costs ~0.33 of recall.** 0.93 same-domain against 0.60 cross-domain. The
   powered number is the optimistic one, and a realistic expectation for the shipped pipeline is
   nearer 0.6 than 0.9.
2. **Failure is bimodal.** r@5 equals r@1 exactly: when the right brand is not the top hit, it is
   not in the top five either. That is good news for gating — a similarity threshold should
   separate hits from misses cleanly, rather than having to reason about a long tail.
3. **KIA fails at 0.00 for both models, and it is not the embedder's fault.** The pool's KIA
   references are all the *old oval logo*; the detected crops are the 2021 rebrand wordmark.
   They are visually different marks. **Retrieval quality is bounded by pool freshness, and no
   encoder choice fixes a stale reference.** This is worth more attention than the model
   comparison it turned up in.

## A claim that did not survive

The README argued that NaFlex must beat `-384` because a fixed square resize squashes wide
wordmarks, and that this is the payload of the index. The `wide` column tests exactly that —
queries with aspect ratio >= 2.5, which is half the pool (median aspect 2.59, p90 5.57, max
10.4), so the test is not toothless.

It did not hold. base-naflex led at 400 brands (0.937 vs 0.932) and large-384 led at 1500 (0.943
vs 0.947). The direction flipped, so it is noise. **The aspect-squash argument is not supported
at retrieval level** and has been softened in the README from a settled reason to an untested
mechanism. base-naflex still wins — on cost, which is measured.

## Reproducing

```bash
bash eval/experiments/08_embedders/run.sh        # pool -> pool, powered
bash eval/experiments/08_embedders/run_crop.sh   # crop -> pool, realistic but n=15
```
