"""
travel_core.py - Data structures and algorithms for the Travel Planner.

Everything here is implemented from scratch (no networkx / heapq):
    Graph        - weighted undirected graph using adjacency lists (dict of dicts)
    MinHeap      - binary-heap priority queue (used by Dijkstra)
    Queue/Stack  - used by BFS and DFS
    DisjointSet  - union-find with path compression + union by rank (Kruskal)

Algorithms:
    dijkstra()        shortest route by distance / time / cost
    bfs_fewest_stops()route with the fewest intermediate cities
    all_routes_dfs()  alternative routes using DFS + backtracking
    kruskal_mst()     cheapest road network connecting all cities
    plan_trip()       multi-stop trip ordering (Held-Karp DP, bitmask)
    components()      connectivity check (detects isolated cities)
"""

INF = float("inf")


# ------------------------------------------------------------------ #
#  Basic data structures
# ------------------------------------------------------------------ #
class MinHeap:
    """Binary min-heap storing (priority, tie_breaker, item)."""

    def __init__(self):
        self._a = []
        self._count = 0

    def __len__(self):
        return len(self._a)

    def push(self, priority, item):
        self._count += 1
        self._a.append((priority, self._count, item))
        self._sift_up(len(self._a) - 1)

    def pop(self):
        if not self._a:
            raise IndexError("pop from empty heap")
        top = self._a[0]
        last = self._a.pop()
        if self._a:
            self._a[0] = last
            self._sift_down(0)
        return top[0], top[2]

    def _sift_up(self, i):
        while i > 0:
            parent = (i - 1) // 2
            if self._a[i] < self._a[parent]:
                self._a[i], self._a[parent] = self._a[parent], self._a[i]
                i = parent
            else:
                break

    def _sift_down(self, i):
        n = len(self._a)
        while True:
            left, right, small = 2 * i + 1, 2 * i + 2, i
            if left < n and self._a[left] < self._a[small]:
                small = left
            if right < n and self._a[right] < self._a[small]:
                small = right
            if small == i:
                break
            self._a[i], self._a[small] = self._a[small], self._a[i]
            i = small


class Node:
    __slots__ = ("val", "next")

    def __init__(self, val):
        self.val, self.next = val, None


class Queue:
    """FIFO queue built on a singly linked list (O(1) enqueue/dequeue)."""

    def __init__(self):
        self.head = self.tail = None
        self.size = 0

    def __len__(self):
        return self.size

    def enqueue(self, val):
        node = Node(val)
        if self.tail:
            self.tail.next = node
        else:
            self.head = node
        self.tail = node
        self.size += 1

    def dequeue(self):
        if not self.head:
            raise IndexError("dequeue from empty queue")
        node = self.head
        self.head = node.next
        if not self.head:
            self.tail = None
        self.size -= 1
        return node.val


class Stack:
    """LIFO stack built on a singly linked list."""

    def __init__(self):
        self.top = None
        self.size = 0

    def __len__(self):
        return self.size

    def push(self, val):
        node = Node(val)
        node.next = self.top
        self.top = node
        self.size += 1

    def pop(self):
        if not self.top:
            raise IndexError("pop from empty stack")
        node = self.top
        self.top = node.next
        self.size -= 1
        return node.val


class DisjointSet:
    """Union-Find with path compression and union by rank."""

    def __init__(self, items):
        self.parent = {x: x for x in items}
        self.rank = {x: 0 for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


# ------------------------------------------------------------------ #
#  Graph
# ------------------------------------------------------------------ #
CRITERIA = {"Distance (km)": "km", "Time (hours)": "hrs", "Cost (Rs)": "cost"}


class Graph:
    """Undirected weighted graph. Each road stores km, hrs and cost."""

    def __init__(self):
        self.adj = {}        # city -> {neighbour: {"km":..,"hrs":..,"cost":..}}
        self.coords = {}     # city -> (lat, lon)
        self.closed = set()  # frozenset({a, b}) of closed roads

    # ---- building ----
    def add_city(self, name, lat, lon):
        self.adj.setdefault(name, {})
        self.coords[name] = (lat, lon)

    def add_road(self, a, b, km, hrs, cost):
        road = {"km": km, "hrs": hrs, "cost": cost}
        self.adj[a][b] = road
        self.adj[b][a] = road

    # ---- queries ----
    def cities(self):
        return sorted(self.adj)

    def is_open(self, a, b):
        return frozenset((a, b)) not in self.closed

    def neighbors(self, u):
        """Yield (city, road) for every OPEN road from u."""
        for v, road in self.adj[u].items():
            if self.is_open(u, v):
                yield v, road

    def roads(self):
        """Every open road once as (a, b, road)."""
        seen = set()
        for a in self.adj:
            for b, road in self.adj[a].items():
                key = frozenset((a, b))
                if key not in seen and key not in self.closed:
                    seen.add(key)
                    yield a, b, road

    def all_roads(self):
        """Every road once, including closed ones."""
        seen = set()
        for a in self.adj:
            for b, road in self.adj[a].items():
                key = frozenset((a, b))
                if key not in seen:
                    seen.add(key)
                    yield a, b, road


# ------------------------------------------------------------------ #
#  Algorithms
# ------------------------------------------------------------------ #
def dijkstra_all(g, src, crit):
    """Shortest distance from src to every city. Returns (dist, prev)."""
    dist = {c: INF for c in g.adj}
    prev = {}
    dist[src] = 0
    heap = MinHeap()
    heap.push(0, src)
    while len(heap):
        d, u = heap.pop()
        if d > dist[u]:
            continue  # stale heap entry
        for v, road in g.neighbors(u):
            nd = d + road[crit]
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heap.push(nd, v)
    return dist, prev


def build_path(prev, src, dst):
    path, cur = [dst], dst
    while cur != src:
        if cur not in prev:
            return []
        cur = prev[cur]
        path.append(cur)
    return path[::-1]


def dijkstra(g, src, dst, crit):
    dist, prev = dijkstra_all(g, src, crit)
    if dist[dst] == INF:
        return INF, []
    return dist[dst], build_path(prev, src, dst)


def bfs_fewest_stops(g, src, dst):
    """Route with the fewest roads (unweighted BFS using our Queue)."""
    q = Queue()
    q.enqueue(src)
    parent = {src: None}
    while len(q):
        u = q.dequeue()
        if u == dst:
            break
        for v, _ in g.neighbors(u):
            if v not in parent:
                parent[v] = u
                q.enqueue(v)
    if dst not in parent:
        return []
    path, cur = [], dst
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    return path[::-1]


def path_totals(g, path):
    t = {"km": 0, "hrs": 0.0, "cost": 0}
    for a, b in zip(path, path[1:]):
        r = g.adj[a][b]
        t["km"] += r["km"]
        t["hrs"] += r["hrs"]
        t["cost"] += r["cost"]
    t["hrs"] = round(t["hrs"], 1)
    return t


def all_routes_dfs(g, src, dst, crit, max_hops=8, limit=500):
    """All simple routes src->dst using an explicit-stack DFS (backtracking)."""
    routes = []
    stack = Stack()
    stack.push((src, [src]))
    while len(stack) and len(routes) < limit:
        u, path = stack.pop()
        if u == dst:
            routes.append(path)
            continue
        if len(path) - 1 >= max_hops:
            continue
        for v, _ in g.neighbors(u):
            if v not in path:
                stack.push((v, path + [v]))
    scored = [(path_totals(g, p)[crit], p) for p in routes]
    scored.sort(key=lambda x: x[0])
    return scored


def kruskal_mst(g, crit):
    """Cheapest set of roads connecting every reachable city (min spanning forest)."""
    edges = sorted(((r[crit], a, b) for a, b, r in g.roads()))
    ds = DisjointSet(g.adj.keys())
    chosen, total = [], 0
    for w, a, b in edges:
        if ds.union(a, b):
            chosen.append((a, b, w))
            total += w
    return chosen, total


def components(g):
    """Connected components via BFS."""
    seen, comps = set(), []
    for start in g.cities():
        if start in seen:
            continue
        comp, q = [], Queue()
        q.enqueue(start)
        seen.add(start)
        while len(q):
            u = q.dequeue()
            comp.append(u)
            for v, _ in g.neighbors(u):
                if v not in seen:
                    seen.add(v)
                    q.enqueue(v)
        comps.append(sorted(comp))
    return comps


def plan_trip(g, start, stops, crit, round_trip=True):
    """
    Best order to visit all `stops` starting from `start` (Held-Karp DP).
    Pairwise costs come from Dijkstra, so the legs follow real roads.
    Returns (total_cost, ordered_cities, full_path) or (INF, [], []).
    """
    stops = [s for s in dict.fromkeys(stops) if s != start]
    if not stops:
        return 0, [start], [start]
    if len(stops) > 10:
        raise ValueError("Please choose at most 10 stops.")

    nodes = [start] + stops
    n = len(stops)
    d = [[INF] * (n + 1) for _ in range(n + 1)]
    prevs = []
    for i, c in enumerate(nodes):
        dist, prev = dijkstra_all(g, c, crit)
        prevs.append(prev)
        for j, c2 in enumerate(nodes):
            d[i][j] = dist[c2]
    if any(d[0][j] == INF for j in range(1, n + 1)):
        return INF, [], []

    full = (1 << n) - 1
    dp = [[INF] * n for _ in range(1 << n)]
    parent = [[-1] * n for _ in range(1 << n)]
    for j in range(n):
        dp[1 << j][j] = d[0][j + 1]
    for mask in range(1, 1 << n):
        for j in range(n):
            if not (mask >> j) & 1 or dp[mask][j] == INF:
                continue
            for k in range(n):
                if (mask >> k) & 1:
                    continue
                nm = mask | (1 << k)
                cand = dp[mask][j] + d[j + 1][k + 1]
                if cand < dp[nm][k]:
                    dp[nm][k] = cand
                    parent[nm][k] = j

    best, last = INF, -1
    for j in range(n):
        tail = d[j + 1][0] if round_trip else 0
        if dp[full][j] + tail < best:
            best, last = dp[full][j] + tail, j
    if best == INF:
        return INF, [], []

    order_idx, mask, j = [], full, last
    while j != -1:
        order_idx.append(j)
        pj = parent[mask][j]
        mask ^= 1 << j
        j = pj
    order = [start] + [stops[i] for i in reversed(order_idx)]
    if round_trip:
        order.append(start)

    full_path = [order[0]]
    for a, b in zip(order, order[1:]):
        leg = build_path(prevs[nodes.index(a)], a, b)
        full_path.extend(leg[1:])
    return best, order, full_path


# ------------------------------------------------------------------ #
#  Sample data: South Indian road network (approximate values)
# ------------------------------------------------------------------ #
CITIES = {
    "Chennai": (13.0827, 80.2707), "Bengaluru": (12.9716, 77.5946),
    "Hyderabad": (17.3850, 78.4867), "Coimbatore": (11.0168, 76.9558),
    "Madurai": (9.9252, 78.1198), "Tiruchirappalli": (10.7905, 78.7047),
    "Puducherry": (11.9416, 79.8083), "Kochi": (9.9312, 76.2673),
    "Thiruvananthapuram": (8.5241, 76.9366), "Mysuru": (12.2958, 76.6394),
    "Vijayawada": (16.5062, 80.6480), "Tirupati": (13.6288, 79.4192),
    "Salem": (11.6643, 78.1460), "Kanyakumari": (8.0883, 77.5385),
}

# (city A, city B, km, hours, toll in Rs). Cost = 5 Rs/km fuel + toll.
ROADS = [
    ("Chennai", "Puducherry", 150, 3.2, 100), ("Chennai", "Tirupati", 135, 2.8, 200),
    ("Chennai", "Bengaluru", 345, 6.0, 600), ("Chennai", "Salem", 340, 5.8, 450),
    ("Chennai", "Tiruchirappalli", 330, 5.5, 400), ("Chennai", "Vijayawada", 450, 7.5, 500),
    ("Puducherry", "Tiruchirappalli", 190, 3.8, 150), ("Tiruchirappalli", "Madurai", 135, 2.5, 150),
    ("Tiruchirappalli", "Salem", 140, 2.7, 200), ("Salem", "Coimbatore", 165, 3.0, 250),
    ("Salem", "Bengaluru", 205, 3.7, 350), ("Bengaluru", "Mysuru", 145, 3.0, 250),
    ("Bengaluru", "Tirupati", 250, 4.7, 300), ("Bengaluru", "Hyderabad", 570, 9.5, 900),
    ("Mysuru", "Coimbatore", 195, 4.5, 100), ("Coimbatore", "Kochi", 190, 4.0, 150),
    ("Coimbatore", "Madurai", 210, 4.0, 200), ("Madurai", "Kanyakumari", 245, 4.5, 150),
    ("Madurai", "Thiruvananthapuram", 300, 5.5, 150), ("Kochi", "Thiruvananthapuram", 205, 4.5, 100),
    ("Kanyakumari", "Thiruvananthapuram", 90, 2.0, 0), ("Hyderabad", "Vijayawada", 275, 4.5, 400),
    ("Tirupati", "Vijayawada", 390, 6.8, 300),
]


def build_sample_graph():
    g = Graph()
    for name, (lat, lon) in CITIES.items():
        g.add_city(name, lat, lon)
    for a, b, km, hrs, toll in ROADS:
        g.add_road(a, b, km, hrs, km * 5 + toll)
    return g