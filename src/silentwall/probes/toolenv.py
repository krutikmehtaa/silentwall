"""Offline tool environment for the tool-mediated probe family.

The point of this family is that an agent can rebuild information it was told to
forget by going and looking it up. So the document store deliberately contains the
private artifacts, and the question is whether a containment method stops the agent
from reaching them.

Everything is local. No network, no subprocess, no eval of untrusted input. The
calculator is a whitelisted AST walk rather than eval, because an agent will
eventually emit something strange and a research harness should not be the thing
that runs it.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from ..hashing import hash_obj

__all__ = ["RetrievedDoc", "DocStore", "safe_calc", "ToolEnv"]

_TOKEN_RE = re.compile(r"[a-z0-9$.%]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass(frozen=True, slots=True)
class RetrievedDoc:
    doc_id: str
    text: str
    score: float
    entity_id: str | None = None
    is_private: bool = False


@dataclass
class DocStore:
    """Tiny BM25 index. Small enough to be exact, which is what we want in a harness."""

    docs: list[RetrievedDoc] = field(default_factory=list)
    k1: float = 1.5
    b: float = 0.75
    _df: dict[str, int] = field(default_factory=dict, repr=False)
    _tf: list[dict[str, int]] = field(default_factory=list, repr=False)
    _len: list[int] = field(default_factory=list, repr=False)

    def add(
        self, doc_id: str, text: str, entity_id: str | None = None, is_private: bool = False
    ) -> None:
        self.docs.append(RetrievedDoc(doc_id, text, 0.0, entity_id, is_private))
        toks = _tokens(text)
        tf: dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        self._tf.append(tf)
        self._len.append(len(toks))
        for t in tf:
            self._df[t] = self._df.get(t, 0) + 1

    def search(self, query: str, top_k: int = 3) -> list[RetrievedDoc]:
        if not self.docs:
            return []
        q = _tokens(query)
        n = len(self.docs)
        avg = sum(self._len) / n if n else 1.0
        scored: list[RetrievedDoc] = []

        for i, doc in enumerate(self.docs):
            tf = self._tf[i]
            dl = self._len[i] or 1
            score = 0.0
            for term in q:
                f = tf.get(term, 0)
                if not f:
                    continue
                df = self._df.get(term, 1)
                idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
                score += (
                    idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / avg))
                )
            if score > 0:
                scored.append(
                    RetrievedDoc(doc.doc_id, doc.text, score, doc.entity_id, doc.is_private)
                )

        scored.sort(key=lambda d: (-d.score, d.doc_id))
        return scored[:top_k]

    def fingerprint(self) -> str:
        """Content hash of the store, so tool state participates in cache keys."""
        return hash_obj([(d.doc_id, d.text, d.is_private) for d in self.docs])


_BIN_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def safe_calc(expr: str) -> float:
    """Evaluate arithmetic without eval.

    Whitelisted node types only. Anything else raises, which is the correct answer
    when a language model hands you a string it wants executed.
    """
    if len(expr) > 200:
        raise ValueError("expression too long")

    def walk(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, int | float):
                raise ValueError("only numeric literals are allowed")
            return float(node.value)
        if isinstance(node, ast.BinOp):
            op = _BIN_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"operator {type(node.op).__name__} not allowed")
            left, right = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Pow) and (abs(right) > 8 or abs(left) > 1e6):
                raise ValueError("exponent out of range")
            return float(op(left, right))
        if isinstance(node, ast.UnaryOp):
            op_u = _UNARY_OPS.get(type(node.op))
            if op_u is None:
                raise ValueError("unary operator not allowed")
            return float(op_u(walk(node.operand)))
        raise ValueError(f"node {type(node).__name__} not allowed")

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"could not parse expression: {exc}") from exc
    return walk(tree)


class ToolEnv:
    """The tools an agent can call, plus a log of what it called.

    The log matters as much as the results: toolcall_count, abstain rate and name
    entropy are three of the twelve behavioural features, so a containment method
    that quietly reroutes tool use shows up in the detector.
    """

    def __init__(self, store: DocStore, top_k: int = 3) -> None:
        self.store = store
        self.top_k = top_k
        self.calls: list[str] = []

    def reset(self) -> None:
        self.calls.clear()

    def search_docs(self, query: str) -> list[RetrievedDoc]:
        self.calls.append("search_docs")
        return self.store.search(query, self.top_k)

    def calc(self, expr: str) -> str:
        self.calls.append("calc")
        try:
            return f"{safe_calc(expr):.6g}"
        except ValueError as exc:
            return f"error: {exc}"

    def dispatch(self, name: str, arg: str) -> str:
        if name == "search_docs":
            docs = self.search_docs(arg)
            return "\n".join(f"[{d.doc_id}] {d.text[:400]}" for d in docs) or "no documents found"
        if name == "calc":
            return self.calc(arg)
        self.calls.append(f"unknown:{name}")
        return f"error: no tool named {name}"

    def state_hash(self, exposed: Sequence[str]) -> str:
        """Part of the cache key. Different tool state means a different question."""
        return hash_obj(self.store.fingerprint(), sorted(exposed), self.top_k)
