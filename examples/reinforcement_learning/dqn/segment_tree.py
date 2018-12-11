import operator


class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element):
        r"""Define a segment tree data structure. 
        
        The major difference with regular array:
        * Update value: slightly slower, O(log capacity) < O(1)
        * Query over subsequence: efficient
        
        Args:
            capacity (int): total size of the array, must be a power of two. 
            operation (operation): mathematical operation e.g. min, max, sum
            neutral_element (obj): neutral element for the operation
                e.g. -inf for max, and 0 for sum. 
        """
        assert capacity > 0 and capacity & (capacity - 1) == 0, 'capacity must be positive and a power of 2.'
        self.capacity = capacity
        self.operation = operation
        self.value = [neutral_element for _ in range(2*capacity)]
        
    def _reducing(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self.value[node]
        mid = (node_start + node_end)//2
        if end <= mid:
            return self._reducing(start, end, 2*node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reducing(start, end, 2*node + 1, mid + 1, node_end)
            else:
                return self.operation(self._reducing(start, mid, 2*node, node_start, mid), 
                                      self._reducing(mid + 1, end, 2*node + 1, mid + 1, node_end))
        
    def reduce(self, start=0, end=None):
        r"""Apply the operation to a contiguous subsequence of the array. 
        
        i.e. operation(arr[start], operation(arr[start+1], operation(... arr[end])))
        
        Args:
            start (int): beginning of the subsequences
            end (int): end of the subsequences
            
        Returns
        -------
        out : obj
            computed result
        """
        if end is None:
            env = self.capacity
        if end < 0:
            end += self.capacity
        return self._reducing(start, end, 1, 0, self.capacity - 1)
    
    def __setitem__(self, index, value):
        # index of the leaf
        index += self.capacity
        self.value[index] = value
        index //= 2
        while index >= 1:
            self.value[index] = self.operation(self.value[2*index], self.value[2*index + 1])
            index //= 2
            
    def __getitem__(self, index):
        assert index >= 0 and index < self.capacity
        return self.value[self.capacity + index]


class SumSegmentTree(SegmentTree):
    r"""Define a sum tree data structure for storing reply priorities. 
    
    A sum tree is a complete binary tree whose leaves contain priority values. 
    Intermediate nodes maintain the sum of the priorities of all leaf nodes in their subtree. 
    
    e.g.
    
    For capacity = 4, the tree may look like this:
                   +---+
                   |2.5|
                   +-+-+
                     |
             +-------+--------+
             |                |
           +-+-+            +-+-+
           |1.5|            |1.0|
           +-+-+            +-+-+
             |                |
        +----+----+      +----+----+
        |         |      |         |
      +-+-+     +-+-+  +-+-+     +-+-+
      |0.5|     |1.0|  |0.5|     |0.5|
      +---+     +---+  +---+     +---+
    """
    def __init__(self, capacity):
        super().__init__(capacity, operator.add, 0.0)
        
    def sum(self, start=0, end=None):
        return super().reduce(start, end)
    
    def find_prefixsum_idx(self, prefixsum):
        r"""Find the highest index `i` in the array such that
        
        sum(arr[0] + arr[1] + ... + arr[i - 1]) <= prefixsum
        
        Args:
            prefixsum (float): upper-bound on the sum of array prefix
            
        Returns
        -------
        out : int
            highest index satisfying the condition
        """
        assert prefixsum >= 0 and prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self.capacity:  # while non-leaf
            if self.value[2*idx] > prefixsum:
                idx = 2*idx
            else:
                prefixsum -= self.value[2*idx]
                idx = 2*idx + 1
        return idx - self.capacity
    
    
class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super().__init__(capacity, min, float('inf'))
        
    def min(self, start=0, end=None):
        return super().reduce(start, end)
