"""End-to-end deterministic baseline and five-node LangGraph verifier."""
import json
import re
import time
from pathlib import Path
from typing import TypedDict

import yaml
from langgraph.graph import StateGraph, END
from sentence_transformers import CrossEncoder
from src.retrieval.bm25 import BM25Retriever, tokenize

class State(TypedDict, total=False):
    claim: str
    analysis: dict
    papers: list
    evidence: list
    verdict: str
    confidence: float
    report: dict
    retries: int
    errors: list

class Verifier:
    def __init__(self, mode='baseline'):
        assert mode in ('baseline', 'agent')
        self.mode = mode
        self.cfg = yaml.safe_load(Path('configs/verification.yaml').read_text())
        self.bm25 = BM25Retriever()
        self.documents = {str(d['doc_id']): d for d in self.bm25.documents}
        self.nli = CrossEncoder(self.cfg['nli_model'])
        self.hybrid = None
        if mode == 'agent':
            from src.retrieval.hybrid import HybridRetriever
            self.hybrid = HybridRetriever()
        graph = StateGraph(State)
        for name, fn in [('analyze', self.analyze), ('retrieve', self.retrieve), ('select', self.select), ('verify', self.verify), ('report', self.report)]:
            graph.add_node(name, fn)
        graph.set_entry_point('analyze')
        graph.add_edge('analyze', 'retrieve')
        graph.add_edge('retrieve', 'select')
        graph.add_edge('select', 'verify')
        graph.add_conditional_edges('verify', self.route, {'retry': 'retrieve', 'done': 'report'})
        graph.add_edge('report', END)
        self.graph = graph.compile()

    def analyze(self, state):
        claim = state['claim'].strip()
        if not claim:
            raise ValueError('Claim must be nonempty')
        return {'analysis': {'normalized_claim': claim, 'keywords': tokenize(claim)}, 'retries': state.get('retries', 0)}

    def retrieve(self, state):
        k = int(self.cfg['bm25_top_k']) + state.get('retries', 0) * 5
        if self.hybrid:
            papers = self.hybrid.search(state['claim'])
        else:
            papers = self.bm25.search(state['claim'], top_k=k)
        return {'papers': papers, 'retries': state.get('retries', 0)}

    def select(self, state):
        query = set(tokenize(state['claim']))
        candidates = []
        for paper in state['papers']:
            doc = self.documents.get(str(paper['doc_id']))
            if not doc:
                continue
            for idx, sentence in enumerate(doc['sentences']):
                overlap = len(query & set(tokenize(sentence))) / max(1, len(query))
                candidates.append({'doc_id': str(doc['doc_id']), 'sentence_id': idx, 'text': sentence, 'title': doc['title'], 'score': overlap})
        candidates.sort(key=lambda x: x['score'], reverse=True)
        # Preserve diverse papers; evidence positions are zero-indexed SciFact sentence IDs.
        per_doc = {}
        selected = []
        for item in candidates:
            pid = item['doc_id']
            if per_doc.get(pid, 0) >= int(self.cfg['max_sentences_per_paper']):
                continue
            selected.append(item)
            per_doc[pid] = per_doc.get(pid, 0) + 1
            if len(selected) >= int(self.cfg['max_evidence']):
                break
        return {'evidence': selected}

    def verify(self, state):
        evidence = state['evidence']
        if not evidence:
            return {'verdict': 'INSUFFICIENT EVIDENCE', 'confidence': 0.0, 'retries': state.get('retries', 0) + 1}
        # CrossEncoder NLI outputs are ordered according to model.config.id2label.
        pairs = [(item['text'], state['claim']) for item in evidence]
        import numpy as np
        logits = np.asarray(self.nli.predict(pairs, apply_softmax=True))
        labels = {int(k): str(v).lower() for k, v in self.nli.model.config.id2label.items()}
        if not any('entail' in v for v in labels.values()) or not any('contrad' in v for v in labels.values()):
            raise ValueError(f'NLI model label mapping unavailable: {labels}')
        ranked = []
        for i, row in enumerate(logits):
            for j, score in enumerate(row):
                label = labels[j]
                if 'entail' in label:
                    ranked.append((float(score), 'SUPPORT', i))
                elif 'contrad' in label:
                    ranked.append((float(score), 'CONTRADICT', i))
        best = max(ranked, default=(0.0, 'INSUFFICIENT EVIDENCE', 0))
        verdict = best[1] if best[0] >= float(self.cfg['min_confidence']) else 'INSUFFICIENT EVIDENCE'
        return {'verdict': verdict, 'confidence': best[0], 'retries': state.get('retries', 0) + (1 if verdict == 'INSUFFICIENT EVIDENCE' and self.mode == 'agent' else 0)}

    def route(self, state):
        if self.mode == 'agent' and state['verdict'] == 'INSUFFICIENT EVIDENCE' and state.get('retries', 0) <= int(self.cfg['max_retries']):
            return 'retry'
        return 'done'

    def report(self, state):
        return {'report': {'claim': state['claim'], 'verdict': state['verdict'], 'confidence': state['confidence'], 'evidence': state['evidence'], 'paper_ids': [str(p['doc_id']) for p in state['papers']], 'retry_count': state.get('retries', 0), 'explanation': 'Verdict is based on the strongest NLI evidence score; it is not scientific consensus.'}}

    def run(self, claim):
        start = time.perf_counter()
        # Use bounded manual retry around the graph to avoid an unbounded conditional loop.
        result = self.graph.invoke({'claim': claim, 'retries': 0})
        report = result['report']
        report['latency_seconds'] = time.perf_counter() - start
        report['tool_calls'] = 2 + (1 if self.hybrid else 0)
        report['llm_tokens'] = 0
        return report

    def close(self):
        if self.hybrid:
            self.hybrid.close()
