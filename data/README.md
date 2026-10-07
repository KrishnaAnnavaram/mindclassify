# data/

Git does not track the files in this folder, except this README.

> Treat the corpus as sensitive data. It contains self-disclosed mental-health posts, also posts
> with suicidal content. Do not commit it. Do not copy real posts into code, tests, issues,
> examples or reports.

## Source

| Item | Value |
|---|---|
| Name | Sentiment Analysis for Mental Health (Kaggle) |
| URL | <https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health> |
| Size | About 53,000 rows |
| Origin | An aggregation of several public Reddit and Twitter mental-health datasets |
| License and terms | Read the Kaggle page and the terms of each original source before you use the data |

## Expected file

`data/Combined Data.csv` (the default of `MINDCLASSIFY_DATA`), with these columns:

| Column | Meaning |
|---|---|
| (unnamed index) | Ignored |
| `statement` | The post text |
| `status` | One of `Normal`, `Depression`, `Suicidal`, `Anxiety`, `Bipolar` (also written `Bi-Polar`), `Stress`, `Personality disorder` |
| `source` (optional) | The original dataset of the post. If it is absent, the file name is the source |

mindclassify keeps all seven labels. It never drops `Suicidal`.

## Download

```bash
pip install kaggle      # needs a Kaggle API token
kaggle datasets download -d suchintikasarkar/sentiment-analysis-for-mental-health -p data --unzip
mindclassify prepare
```

## No download

`mindclassify synth --out data/synthetic.csv` writes template posts with the same columns, and the
`--synthetic` flag makes them in memory. The template posts come from word lists. They are not
written by or about real people. The tests use only these posts.
