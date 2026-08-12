class UF:
    """
    一個基本的並查集 (Union-Find) 實現，用於高效管理不相交集合。
    此實現適用於 M3W 聚類算法中基於連通性的核心點合併步驟。
    """

    def __init__(self, n):
        """
        初始化並查集。

        參數:
            n (int): 數據點的總數。每個點的初始索引 (0 到 n-1) 將作為一個獨立的集合。
        """
        # _id[i] 表示點 i 的父節點。如果 _id[i] == i，則 i 是該集合的代表元（根）。
        self._id = list(range(n))
        # _sz[i] 表示以 i 為根的集合的大小（僅在 i 是根時有效）。用於按秩合併。
        self._sz = [1] * n
        # 當前集合（簇）的數量
        self._count = n

    def find(self, p):
        """
        查找點 p 所屬集合的代表元（根節點），並進行路徑壓縮。

        參數:
            p (int): 數據點的索引。

        返回:
            int: 點 p 所屬集合的代表元索引。
        """
        # 路徑壓縮：在查找根節點的同時，將路徑上的所有節點直接指向根。
        while p != self._id[p]:
            self._id[p] = self._id[self._id[p]]  # 路徑壓縮（隔代壓縮，非完全壓縮）
            p = self._id[p]
        return p

    def union(self, p, q):
        """
        如果點 p 和點 q 不屬於同一個集合，則將它們所在的集合合併。

        參數:
            p (int): 第一個點的索引。
            q (int): 第二個點的索引。

        返回:
            bool: 如果成功進行了合併操作則返回 True；如果 p 和 q 已經在同一個集合中則返回 False。
        """
        rootP = self.find(p)
        rootQ = self.find(q)
        if rootP == rootQ:
            return False  # 已經在同一個集合中，無需合併

        # 按秩合併：將較小的樹連接到較大的樹下
        if self._sz[rootP] < self._sz[rootQ]:
            self._id[rootP] = rootQ
            self._sz[rootQ] += self._sz[rootP]
        else:
            self._id[rootQ] = rootP
            self._sz[rootP] += self._sz[rootQ]
        self._count -= 1
        return True

    def connected(self, p, q):
        """
        檢查點 p 和點 q 是否屬於同一個集合。

        參數:
            p (int): 第一個點的索引。
            q (int): 第二個點的索引。

        返回:
            bool: 如果屬於同一個集合則返回 True，否則返回 False。
        """
        return self.find(p) == self.find(q)

    @property
    def count(self):
        """
        返回當前集合（簇）的數量。

        返回:
            int: 當前集合的數量。
        """
        return self._count