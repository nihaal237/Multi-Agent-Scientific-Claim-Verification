"""Evaluate end-to-end claim verification on a deterministic dev subset."""
import argparse
import json
from pathlib import Path
from statistics import mean
from sklearn.metrics import f1_score
from src.agents.workflow import Verifier

def evaluate(mode, limit):
    claims = [json.loads(x) for x in Path('data/processed/claims_dev.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    # Only labeled claims: absence of annotations is NOT an explicit negative label.
    claims = [c for c in claims if c.get('evidence')]
    claims = sorted(claims, key=lambda c: str(c['id']))[:limit]
    verifier = Verifier(mode)
    rows = []
    try:
        for claim in claims:
            gold_docs, gold_sentences, labels = set(), set(), set()
            for doc_id, entries in claim['evidence'].items():
                gold_docs.add(str(doc_id))
                for entry in entries:
                    labels.add(entry['label'])
                    for sid in entry.get('sentences', []):
                        gold_sentences.add((str(doc_id), int(sid)))
            if len(labels) != 1:
                continue  # ambiguous multi-stance cases excluded, counted below
            pred = verifier.run(claim['claim'])
            pred_docs = set(pred['paper_ids'][:10])
            pred_sentences = {(str(x['doc_id']), int(x['sentence_id'])) for x in pred['evidence']}
            tp = len(pred_sentences & gold_sentences)
            precision = tp / len(pred_sentences) if pred_sentences else 0
            recall = tp / len(gold_sentences) if gold_sentences else 0
            rows.append({'id': claim['id'], 'gold_label': next(iter(labels)), 'pred_label': pred['verdict'], 'recall_at_5': len(set(pred['paper_ids'][:5]) & gold_docs)/len(gold_docs), 'recall_at_10': len(pred_docs & gold_docs)/len(gold_docs), 'evidence_precision': precision, 'evidence_recall': recall, 'evidence_f1': 2*precision*recall/(precision+recall) if precision+recall else 0, 'latency_seconds': pred['latency_seconds'], 'tool_calls': pred['tool_calls'], 'llm_tokens': pred['llm_tokens'], 'joint_correct': int(pred['verdict'] == next(iter(labels)) and tp > 0)})
    finally:
        verifier.close()
    if not rows:
        raise RuntimeError('No eligible labeled examples')
    y_true = [r['gold_label'] for r in rows]
    y_pred = [r['pred_label'] for r in rows]
    metrics = {'examples': len(rows), 'excluded_ambiguous': len(claims)-len(rows), 'recall_at_5': mean(r['recall_at_5'] for r in rows), 'recall_at_10': mean(r['recall_at_10'] for r in rows), 'evidence_precision': mean(r['evidence_precision'] for r in rows), 'evidence_recall': mean(r['evidence_recall'] for r in rows), 'evidence_f1': mean(r['evidence_f1'] for r in rows), 'label_macro_f1': f1_score(y_true, y_pred, labels=['SUPPORT','CONTRADICT'], average='macro', zero_division=0), 'joint_correctness': mean(r['joint_correct'] for r in rows), 'mean_latency_seconds': mean(r['latency_seconds'] for r in rows), 'mean_tool_calls': mean(r['tool_calls'] for r in rows), 'total_llm_tokens': sum(r['llm_tokens'] for r in rows)}
    Path('reports').mkdir(exist_ok=True)
    Path(f'reports/{mode}_end_to_end_metrics.json').write_text(json.dumps(metrics, indent=2))
    Path(f'reports/{mode}_predictions.json').write_text(json.dumps(rows, indent=2))
    print(json.dumps(metrics, indent=2))
    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['baseline','agent'], required=True)
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum number of labeled claims to evaluate (default: all)'
    )
    args = parser.parse_args()
    evaluate(args.mode, args.limit)
