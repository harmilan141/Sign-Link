import sacrebleu.metrics
import inspect
print("BLEU class signature:", inspect.signature(sacrebleu.metrics.BLEU))
# Test running with BLEU class
bleu_1 = sacrebleu.metrics.BLEU(max_ngram_order=1).corpus_score(['hello world'], [['hello world']]).score
print("BLEU-1 score:", bleu_1)
