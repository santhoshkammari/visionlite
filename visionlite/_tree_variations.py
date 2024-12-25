from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional
import json
import hashlib

@dataclass
class TreeNode:
    id: int
    query: str
    score: float
    children: List['TreeNode'] = field(default_factory=list)
    parent_query: Optional[str] = ""


class Tree:
    def __init__(self, max_depth: int, gen_vars_func, score_func, cache: bool = True):
        self.max_depth = max_depth
        self.root = None
        self.gen_vars_func = gen_vars_func
        self.score_func = score_func
        self.node_id_counter = 0
        self.node_id_map: Dict[int, TreeNode] = {}
        self._cache = cache
        self._cache_dir = Path('tree_cache')
        self._cache_dir.mkdir(exist_ok=True)


    @property
    def total_nodes(self) -> int:
        '''perform dfs and get nodes count'''

        def dfs_count(node: TreeNode) -> int:
            if not node:
                return 0
            return 1 + sum(dfs_count(child) for child in node.children)

        return dfs_count(self.root) if self.root else 0

    @property
    def height(self) -> int:
        '''Get the maximum height/depth of the tree'''

        def get_height(node: TreeNode) -> int:
            if not node:
                return 0
            if not node.children:
                return 1
            return 1 + max(get_height(child) for child in node.children)

        return get_height(self.root) if self.root else 0


    @property
    def leaf_nodes(self) -> List[TreeNode]:
        '''Get all leaf nodes (nodes with no children)'''
        leaves = []

        def collect_leaves(node: TreeNode):
            if not node:
                return
            if not node.children:
                leaves.append(node)
            for child in node.children:
                collect_leaves(child)

        collect_leaves(self.root)
        return leaves

    @property
    def average_score(self) -> float:
        '''Calculate average score across all nodes'''

        def sum_scores(node: TreeNode) -> tuple[float, int]:
            if not node:
                return 0.0, 0
            score_sum = node.score
            count = 1
            for child in node.children:
                child_sum, child_count = sum_scores(child)
                score_sum += child_sum
                count += child_count
            return score_sum, count

        if not self.root:
            return 0.0
        total_score, total_count = sum_scores(self.root)
        return total_score / total_count if total_count > 0 else 0.0

    @property
    def levels(self) -> Dict[int, List[TreeNode]]:
        '''Get nodes organized by their level in the tree'''
        level_dict = {}

        def collect_levels(node: TreeNode, level: int = 0):
            if not node:
                return
            if level not in level_dict:
                level_dict[level] = []
            level_dict[level].append(node)
            for child in node.children:
                collect_levels(child, level + 1)

        collect_levels(self.root)
        return level_dict

    @property
    def is_empty(self) -> bool:
        '''Check if the tree is empty'''
        return self.root is None

    @property
    def branching_factor(self) -> float:
        '''Calculate average number of children per non-leaf node'''
        total_children = 0
        non_leaf_nodes = 0

        def count_children(node: TreeNode):
            nonlocal total_children, non_leaf_nodes
            if not node:
                return
            if node.children:
                total_children += len(node.children)
                non_leaf_nodes += 1
            for child in node.children:
                count_children(child)

        count_children(self.root)
        return total_children / non_leaf_nodes if non_leaf_nodes > 0 else 0.0

    @property
    def score_distribution(self) -> Dict[float, int]:
        '''Get distribution of scores across all nodes'''
        distribution = {}

        def collect_scores(node: TreeNode):
            if not node:
                return
            score = round(node.score, 2)  # Round to 2 decimal places
            distribution[score] = distribution.get(score, 0) + 1
            for child in node.children:
                collect_scores(child)

        collect_scores(self.root)
        return dict(sorted(distribution.items()))

    @property
    def max_width(self) -> int:
        '''Get the maximum width (number of nodes at any level)'''
        return max(len(nodes) for nodes in self.levels.values()) if not self.is_empty else 0

    @property
    def unique_queries(self) -> set:
        '''Get set of unique queries in the tree'''
        queries = set()

        def collect_queries(node: TreeNode):
            if not node:
                return
            queries.add(node.query)
            for child in node.children:
                collect_queries(child)

        collect_queries(self.root)
        return queries

    @property
    def depth_scores(self) -> Dict[int, float]:
        '''Get average scores at each depth level'''
        scores_by_depth = {}
        counts_by_depth = {}

        def collect_depth_scores(node: TreeNode, depth: int = 0):
            if not node:
                return
            if depth not in scores_by_depth:
                scores_by_depth[depth] = 0.0
                counts_by_depth[depth] = 0
            scores_by_depth[depth] += node.score
            counts_by_depth[depth] += 1
            for child in node.children:
                collect_depth_scores(child, depth + 1)

        collect_depth_scores(self.root)
        return {depth: scores_by_depth[depth] / counts_by_depth[depth]
                for depth in scores_by_depth}

    @property
    def path_lengths(self) -> Dict[str, int]:
        '''Get path lengths from root to each leaf node'''
        paths = {}

        def collect_paths(node: TreeNode, path_length: int = 0):
            if not node:
                return
            if not node.children:  # Leaf node
                paths[node.query] = path_length
            for child in node.children:
                collect_paths(child, path_length + 1)

        collect_paths(self.root)
        return paths

    @property
    def query_complexity(self) -> Dict[str, int]:
        '''Get word count for each unique query'''
        return {query: len(query.split())
                for query in self.unique_queries}

    @property
    def similarity_stats(self) -> Dict[str, float]:
        '''Get statistics about similarity scores'''
        scores = []

        def collect_scores(node: TreeNode):
            if not node:
                return
            scores.append(node.score)
            for child in node.children:
                collect_scores(child)

        collect_scores(self.root)
        if not scores:
            return {}

        return {
            'min_score': min(scores),
            'max_score': max(scores),
            'avg_score': sum(scores) / len(scores),
            'score_range': max(scores) - min(scores)
        }

    def _get_cache_path(self, query: str) -> Path:
        # Create a unique filename based on the query
        query_hash = hashlib.md5(query.encode()).hexdigest()
        return self._cache_dir / f"{query_hash}.json"

    def build(self, query: str):
        if self._cache:
            cache_path = self._get_cache_path(query)
            if cache_path.exists():
                # Load from cache if exists
                loaded_tree = self.from_json(
                    str(cache_path),
                    self.gen_vars_func,
                    self.score_func
                )
                self.root = loaded_tree.root
                self.node_id_counter = loaded_tree.node_id_counter
                self.node_id_map = loaded_tree.node_id_map
                return

        # If no cache or cache disabled, build the tree
        self.root = TreeNode(id=self.get_next_id(), query=query, score=1.0,parent_query=query)
        self.node_id_map[self.root.id] = self.root
        dq = deque([(self.root, 0)])
        while dq:
            node, depth = dq.popleft()
            if depth >= self.max_depth:
                continue
            query_variations = self.gen_vars_func(node.query,ctx=node.parent_query)
            for q in query_variations:
                cn = TreeNode(id=self.get_next_id(), query=q, score=self.score_func(node.query, q),
                              parent_query=node.query)
                node.children.append(cn)
                self.node_id_map[cn.id] = cn
                dq.append((cn, depth + 1))

        # Cache the newly built tree if caching is enabled
        self.remove_duplicates(self.root)
        self.save(str(self._get_cache_path(query)))

    def print_tree(self):
        if not self.root:
            return

        def get_tree_lines(node, prefix="", is_last=True):
            lines = []
            conn = "└── " if is_last else "├── "
            lines.append(f"{prefix}{conn}{node.query} ({node.score:.2f})")
            child_prefix = prefix + ("    " if is_last else "│   ")
            for i, child in enumerate(node.children):
                is_last_child = (i == len(node.children) - 1)
                lines.extend(get_tree_lines(child, child_prefix, is_last_child))
            return lines

        print(f"{self.root.query} ({self.root.score:.2f})")
        for i, child in enumerate(self.root.children):
            is_last = (i == len(self.root.children) - 1)
            lines = get_tree_lines(child, "", is_last)
            print("\n".join(lines))

    def get_next_id(self) -> int:
        self.node_id_counter += 1
        return self.node_id_counter



    def get_node_by_score(self, score: float) -> Optional[TreeNode]:
        def _find_by_score(node: TreeNode) -> Optional[TreeNode]:
            if node.score == score:
                return node
            for child in node.children:
                result = _find_by_score(child)
                if result:
                    return result
            return None

        if self.root:
            return _find_by_score(self.root)
        return None

    def get_node_by_id(self, node_id: int) -> Optional[TreeNode]:
        return self.node_id_map.get(node_id)

    def get_top_n_nodes(
        self,
        top_n: int | None = None
    ) -> List[TreeNode]:
        def _collect_nodes(node: TreeNode) -> List[TreeNode]:
            nodes = [node]
            for child in node.children:
                nodes.extend(_collect_nodes(child))
            return nodes

        all_nodes = _collect_nodes(self.root)
        all_nodes.sort(key=lambda x: x.score, reverse=True)
        return all_nodes[:top_n] if top_n else all_nodes

    def get_top_nodes(self,
                  threshold:float=0.85):
        all_nodes = self.get_top_n_nodes()
        all_nodes = [node for node in all_nodes if node.score >= threshold]
        return all_nodes

    def topk(self,k=3,threshold:Optional[float]=None):
        top_nodes = self.get_top_n_nodes(top_n=k)
        final = []
        for _ in top_nodes:
            if not threshold or _.score >= threshold:
                final.append(_.query)
        return final


    def save(self, filename: str):
        def serialize_node(node: TreeNode) -> dict:
            return {
                'id': node.id,
                'query': node.query,
                'score': node.score,
                'parent_query':node.parent_query,
                'children': [serialize_node(child) for child in node.children]
            }

        Path(filename).write_text(json.dumps(serialize_node(self.root), indent=4))

    @classmethod
    def from_json(cls, filename: str, gen_vars_func, score_func):
        """
        Create a new Tree instance from a JSON file.

        Args:
            filename: Path to the JSON file
            gen_vars_func: Function to generate query variations
            score_func: Function to score query similarities

        Returns:
            Tree: A new Tree instance with the loaded structure
        """
        with open(filename, 'r') as f:
            data = json.load(f)

        def get_max_depth(node_data) -> int:
            if not node_data.get('children'):
                return 0
            return 1 + max(get_max_depth(child) for child in node_data['children'])

        max_depth = get_max_depth(data)

        tree = cls(
            max_depth=max_depth,
            gen_vars_func=gen_vars_func,
            score_func=score_func
        )

        def deserialize_node(node_data: dict) -> TreeNode:
            node = TreeNode(
                id=node_data['id'],
                query=node_data['query'],
                score=node_data['score'],
                parent_query=node_data['parent_query'],
                children=[]
            )
            node.children = [deserialize_node(child) for child in node_data['children']]
            tree.node_id_counter = max(tree.node_id_counter, node.id)
            tree.node_id_map[node.id] = node
            return node

        tree.root = deserialize_node(data)
        tree.node_id_counter += 1

        return tree

    def remove_duplicates(self, root: TreeNode):
        """
        Removes duplicate queries from the tree while keeping the instance with the highest score.

        Args:
            root (TreeNode): The root node of the tree

        The function:
        1. First collects all nodes and keeps track of the highest scoring instance of each query
        2. Then rebuilds the tree connections keeping only the highest scoring instances
        3. Updates the node_id_map to reflect the new structure
        """
        # Step 1: Collect all nodes and find highest scoring instances
        query_to_best_node = {}

        def collect_best_nodes(node):
            if node.query not in query_to_best_node or node.score > query_to_best_node[node.query].score:
                query_to_best_node[node.query] = node
            for child in node.children:
                collect_best_nodes(child)

        collect_best_nodes(root)

        # Step 2: Rebuild tree connections
        seen_queries = set()

        def rebuild_connections(node):
            if node.query in seen_queries:
                return []

            seen_queries.add(node.query)
            new_children = []

            for child in node.children:
                # Get the best instance of this child's query
                best_child = query_to_best_node[child.query]
                if child.query not in seen_queries:
                    # Create a new node with the best score but maintaining the parent relationship
                    new_node = TreeNode(
                        id=best_child.id,
                        query=best_child.query,
                        score=best_child.score,
                        parent_query=node.query
                    )
                    new_node.children = rebuild_connections(best_child)
                    new_children.append(new_node)

            return new_children

        # Rebuild starting from root
        root.children = rebuild_connections(root)

        # Step 3: Update node_id_map
        self.node_id_map.clear()

        def update_id_map(node):
            self.node_id_map[node.id] = node
            for child in node.children:
                update_id_map(child)

        update_id_map(root)

