# Data Directory

Downloaded datasets and generated indexes are stored here locally and ignored by
git.

Expected layout:

```text
data/
  raw/
    nvidia_techqa_rag_eval/
      train.json
      corpus.zip
    primeqa_techqa/
      TechQA.tar.gz
  processed/
  indexes/
```

Do not commit dataset files, extracted corpora, indexes, traces, or model
outputs.
