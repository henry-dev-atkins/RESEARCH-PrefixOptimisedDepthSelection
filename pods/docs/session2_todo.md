# Session 2 TODO

## Goal

Close the broader Session 2 transformer comparator block by finishing:

- RoBERTa main-variant evidence
- final Session 2 interpretation notes

## Current Status

### BERT

`sst2`

- `full`: done
- `pods_all`: done
- `pods_sandwich`: done
- `shallow_d2`: done
- `shallow_d4`: done
- `naive_truncation`: done
- seed `11` for `full` / `pods_all` / `pods_sandwich`: done
- seed `19` for `full` / `pods_all` / `pods_sandwich`: done
- current processed means:
  - `full`: `0.9285168195718654`
  - `pods_all`: `0.8925840978593272`
  - `pods_sandwich`: `0.9032874617737003`
  - `naive_truncation`: `0.9243119266055045`

`mrpc`

- `full`: done
- `pods_all`: done
- `pods_sandwich`: done
- `shallow_d2`: done
- `shallow_d4`: done
- `naive_truncation`: done
- seed `11` for `full` / `pods_all` / `pods_sandwich`: done
- seed `19` for `full` / `pods_all` / `pods_sandwich`: done
- current processed means:
  - `full`: `0.815359477124183`
  - `pods_all`: `0.8120915032679739`
  - `pods_sandwich`: `0.8202614379084967`
  - `naive_truncation`: `0.821078431372549`

`rte`

- `full`: done
- `pods_all`: done
- `pods_sandwich`: done
- `shallow_d2`: done
- `shallow_d4`: done
- `naive_truncation`: done
- seed `11` for `full` / `pods_all` / `pods_sandwich`: done
  - `full`: selected depth `12`, final accuracy `0.6642599277978339`
  - `pods_all`: selected depth `10`, final accuracy `0.6498194945848376`
  - `pods_sandwich`: selected depth `12`, final accuracy `0.6606498194945848`
- seed `19` for `full` / `pods_all` / `pods_sandwich`: done
  - `full`: selected depth `12`, final accuracy `0.592057761732852`
  - `pods_all`: selected depth `6`, final accuracy `0.6353790613718412`
  - `pods_sandwich`: selected depth `6`, final accuracy `0.6462093862815884`
- current processed means:
  - `full`: `0.6353790613718412`
  - `pods_all`: `0.6486161251504212`
  - `pods_sandwich`: `0.6522262334536703`
  - `naive_truncation`: `0.6498194945848376`

### RoBERTa

- code/config scaffold: done in committed code
- behavioural test scaffold: done in committed code
- accepted raw run:
  - `sst2_roberta_base_full`
- remaining main-variant runs:
  - `sst2_roberta_base_pods_all`
  - `sst2_roberta_base_pods_sandwich`
  - `mrpc_roberta_base_full`
  - `mrpc_roberta_base_pods_all`
  - `mrpc_roberta_base_pods_sandwich`
  - `rte_roberta_base_full`
  - `rte_roberta_base_pods_all`
  - `rte_roberta_base_pods_sandwich`

### LayerDrop

- schema/config scaffold: done in committed code
- strict Milestone 3 runs: done
- selected depths and final accuracies:
  - `sst2_bert_base_layerdrop`: depth `6`, accuracy `0.8876146788990825`
  - `mrpc_bert_base_layerdrop`: depth `8`, accuracy `0.7990196078431373`
  - `rte_bert_base_layerdrop`: depth `10`, accuracy `0.631768953068592`

### Milestone 3

- strict `SPEC.md` Milestone 3 bar: complete
- broader Session 2 comparator closure: still pending the remaining RoBERTa main variants

## Immediate Next Actions

1. Run the remaining RoBERTa seed-7 main variants for `sst2`:
   - `sst2_roberta_base_pods_all`
   - `sst2_roberta_base_pods_sandwich`
2. Run RoBERTa seed-7 main variants for `mrpc`:
   - `mrpc_roberta_base_full`
   - `mrpc_roberta_base_pods_all`
   - `mrpc_roberta_base_pods_sandwich`
3. Run RoBERTa seed-7 main variants for `rte`:
   - `rte_roberta_base_full`
   - `rte_roberta_base_pods_all`
   - `rte_roberta_base_pods_sandwich`
4. Refresh processed summaries and frontier plots again once those runs are complete.
5. Write the final Session 2 interpretation once the RoBERTa comparators are present.
