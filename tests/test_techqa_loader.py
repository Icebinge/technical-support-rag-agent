import json
from zipfile import ZipFile

from ts_rag_agent.infrastructure.techqa_loader import (
    compute_dataset_stats,
    list_corpus_filenames,
    load_nvidia_samples,
)


def test_load_and_compute_stats(tmp_path):
    train_json = tmp_path / "train.json"
    corpus_zip = tmp_path / "corpus.zip"

    train_json.write_text(
        json.dumps(
            [
                {
                    "id": "q1",
                    "question": "How do I restart service A?",
                    "answer": "Restart service A with systemctl.",
                    "is_impossible": False,
                    "contexts": [
                        {"filename": "doc1.txt", "text": "Use systemctl restart service-a."}
                    ],
                },
                {
                    "id": "q2",
                    "question": "What is the database password?",
                    "answer": "-",
                    "is_impossible": True,
                    "contexts": [],
                },
            ]
        ),
        encoding="utf-8",
    )

    with ZipFile(corpus_zip, "w") as archive:
        archive.writestr("corpus/doc1.txt", "Use systemctl restart service-a.")

    samples = load_nvidia_samples(train_json)
    corpus = list_corpus_filenames(corpus_zip)
    stats = compute_dataset_stats(samples, corpus)

    assert len(samples) == 2
    assert stats.answerable_rows == 1
    assert stats.impossible_rows == 1
    assert stats.missing_referenced_files == 0
