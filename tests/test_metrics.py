from file_router.evaluation.metrics import answer_metrics, bleu, rouge_l


def test_identical_answer_scores_one():
    metrics = answer_metrics("17", "17")
    assert metrics["exact_match"] == 1.0
    assert metrics["token_f1"] == 1.0
    assert metrics["bleu_1"] == 1.0
    assert metrics["rouge_l"] == 1.0
    assert metrics["numeric_exact"] == 1.0


def test_partial_answer_exposes_precision_and_recall():
    metrics = answer_metrics("beach and forest", "beach mountains forest")
    assert 0.0 < metrics["token_recall"] < 1.0
    assert 0.0 < metrics["token_precision"] < 1.0
    assert 0.0 < bleu("beach and forest", "beach mountains forest", 1) < 1.0
    assert 0.0 < rouge_l("beach forest", "beach mountains forest") < 1.0


def test_chinese_tokenization_is_not_whitespace_dependent():
    metrics = answer_metrics("Beijing", "Beijing")
    assert metrics["token_f1"] == 1.0
    assert metrics["bleu_1"] == 1.0


def test_numeric_formatting_ignores_thousands_separator_styles():
    metrics = answer_metrics("3,02,16,492.00", "30216492.00")
    assert metrics["exact_match"] == 1.0
    assert metrics["numeric_exact"] == 1.0
