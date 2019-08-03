import operator


class SegmentTree(object):
    r"""Defines a segment tree data structure. 
    
    It can be regarded as regular array, but with two major differences
    
    - Value modification is slower: O(ln(capacity)) instead of O(1)
    - Efficient reduce operation over contiguous subarray: O(ln(segment size))
    
    Args:
        capacity (int): total number of elements, it must be a power of two.
        operation (lambda): binary operation forming a group, e.g. sum, min
        identity_element (object): identity element in the group, e.g. 0 for sum
    
    """
    def __init__(self, capacity, operation, identity_element):
        assert capacity > 0 and capacity & (capacity - 1) == 0, 'capacity must be positive and a power of 2.'
        self.capacity = capacity
        self.operation = operation
        self.value = [identity_element for _ in range(2*capacity)]
        
    def _reduce(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self.value[node]
        mid = (node_start + node_end)//2
        if end <= mid:  # go to left child
            return self._reduce(start, end, 2*node, node_start, mid)
        else:
            if start >= mid + 1:  # go to right child
                return self._reduce(start, end, 2*node + 1, mid + 1, node_end)
            else:
                return self.operation(self._reduce(start, mid, 2*node, node_start, mid), 
                                      self._reduce(mid + 1, end, 2*node + 1, mid + 1, node_end))
            
    def reduce(self, start=0, end=None):
        r"""Returns result of operation(A[start], operation(A[start+1], operation(... A[end - 1]))).
        
        Args:
            start (int): start of segment
            end (int): end of segment
            
        Returns:
            object: result of reduce operation
        """
        if end is None:
            end = self.capacity
        if end < 0:
            end += self.capacity
        end -= 1
        return self._reduce(start, end, 1, 0, self.capacity - 1)
    
    def __setitem__(self, index, value):
        # index of leaf
        index += self.capacity
        self.value[index] = value
        index //= 2
        while index >= 1:
            self.value[index] = self.operation(self.value[2*index], self.value[2*index + 1])
            index //= 2
            
    def __getitem__(self, index):
        assert 0 <= index < self.capacity
        return self.value[index + self.capacity]


class SumTree(SegmentTree):
    r"""Defines the sum tree for storing replay priorities. 
    
    Each leaf node contains priority value. Internal nodes maintain the sum of the priorities
    of all leaf nodes in their subtrees. 
    
    """
    def __init__(self, capacity):
        super().__init__(capacity, operator.add, 0.0)
        
    def sum(self, start=0, end=None):
        r"""Return A[start] + ... + A[end - 1]"""
        return super().reduce(start, end)
    
    def find_prefixsum_index(self, prefixsum):
        r"""Find the highest index `i` in the array such that
        sum(A[0] + A[1] + ... + A[i - 1]) <= prefixsum
        
        if array values are probabilities, this function efficiently sample indices according
        to the discrete probability. 
        
        Args:
            prefixsum (float): prefix sum. 
        
        Returns:
            int: highest index satisfying the prefixsum constraint
        """
        assert 0 <= prefixsum <= self.sum() + 1e-5
        index = 1
        while index < self.capacity:  # while non-leaf
            if self.value[2*index] > prefixsum:
                index = 2*index
            else:
                prefixsum -= self.value[2*index]
                index = 2*index + 1
        return index - self.capacity


class MinTree(SegmentTree):
    def __init__(self, capacity):
        super().__init__(capacity, min, float('inf'))
        
    def min(self, start=0, end=None):
        r"""Returns min(A[start], ..., A[end])"""
        return super().reduce(start, end)
