import uuid
from dataclasses import dataclass, field
from operator import attrgetter
from pathlib import Path
from typing import List, Set, Optional, Tuple

from wordllama import WordLlama

@dataclass
class Node:
    id: str
    text: str
    chunk_id: int = -1
    node_score: float = 0.0
    total_score: float = 0.0
    children: List['Node'] = field(default_factory=list)

class WordLlamaRetreiverTree:
    def __init__(self, context: str):
        self.word_llama = WordLlama.load()
        self.chunks: List[str] = []
        self._load_and_process_file(context)


    def retrieve(self, query: str, max_depth: int = 3, max_tot_score: float = 1,
                 k=3):
        self.root = self._build_tree(
            query=query,
            max_depth=max_depth,
            max_tot_score=max_tot_score,
            k=k
        )
        return self._get_tree_text(self.root, query)

    def _load_and_process_file(self,context) -> None:
        """Load and process the input file."""
        self.chunks = self.word_llama.split(context)

    def _generate_node_id(self) -> str:
        """Generate a unique node ID."""
        return str(uuid.uuid4())

    def _build_tree(self, query: str, max_depth: int = 3, max_tot_score: float = 1,
                    k=3) -> Node:
        """Build the knowledge tree starting from the query."""
        root = Node(text=query, id=self._generate_node_id())
        self._build_recursive(
            node=root,
            max_depth=max_depth,
            max_tot_score=max_tot_score,
            depth=0,
            used_chunks=set(),
            k=k
        )
        return root

    def _build_recursive(
        self,
        node: Node,
        max_depth: int,
        max_tot_score: float,
        depth: int,
        used_chunks: Set[int],
        k: int = 3  # Number of children to consider per node (default: 3)
    ) -> None:
        """Recursively build the tree structure."""
        if depth >= max_depth:
            return

        indexed_children = [
            (text, score, idx)
            for idx, (text, score) in enumerate(self.word_llama.rank(
                query=node.text,
                docs=self.chunks,
                sort=False
            ))
        ]
        sorted_children = sorted(
            indexed_children,
            key=lambda x: x[1],
            reverse=True
        )[:k]

        for child_text, score, cid in sorted_children:
            if cid in used_chunks or score < 0.1:
                continue

            child = Node(
                id=self._generate_node_id(),
                text=child_text,
                chunk_id=cid,
                node_score=score
            )
            child.total_score = node.total_score + score

            if child.total_score <= max_tot_score:
                used_chunks.add(cid)
                node.children.append(child)
                self._build_recursive(
                    node=child,
                    max_depth=max_depth,
                    max_tot_score=max_tot_score,
                    depth=depth + 1,
                    used_chunks=used_chunks,
                    k=k  # Number of children to consider per node (default: 3)
                )

    def _get_tree_text(self, root: Node, query: str) -> str:
        """Get the text representation of the tree, excluding the query."""
        visited = set()
        nodes: List[Node] = []

        def _dfs(current_node: Node) -> None:
            if current_node.id in visited:
                return
            visited.add(current_node.id)
            nodes.append(current_node)
            for child in current_node.children:
                _dfs(child)

        _dfs(root)
        nodes = sorted(nodes, key=attrgetter('chunk_id'))
        node_texts = [node.text for node in nodes if node.text != query]
        return '\n'.join(node_texts)
