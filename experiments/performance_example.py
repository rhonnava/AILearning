import time
import sys
import gc
from typing import List, Dict, Any
import threading
import queue

# BUG 1: Inefficient Algorithm - O(n²) instead of O(n log n)
class InefficientSorter:
    """Demonstrates inefficient sorting algorithm."""
    
    # VULNERABLE: Bubble sort - O(n²) complexity
    def bubble_sort_vulnerable(self, arr: List[int]) -> List[int]:
        """VULNERABLE: Using bubble sort which is O(n²)."""
        arr_copy = arr.copy()
        n = len(arr_copy)
        
        for i in range(n):
            for j in range(0, n - i - 1):
                if arr_copy[j] > arr_copy[j + 1]:
                    arr_copy[j], arr_copy[j + 1] = arr_copy[j + 1], arr_copy[j]
        
        return arr_copy
    
    # FIXED: Built-in sort - O(n log n) complexity
    def sort_fixed(self, arr: List[int]) -> List[int]:
        """FIXED: Using Python's built-in sort which is O(n log n)."""
        return sorted(arr)

# BUG 2: Memory Leak - Unclosed Resources
class MemoryLeakExample:
    """Demonstrates memory leaks from unclosed resources."""
    
    def __init__(self):
        self.file_handles = []
        self.threads = []
    
    # VULNERABLE: Memory leak from unclosed files
    def read_files_vulnerable(self, filenames: List[str]) -> List[str]:
        """VULNERABLE: Files are never closed, causing memory leaks."""
        contents = []
        for filename in filenames:
            try:
                # BUG: File handle is never closed
                f = open(filename, 'r')
                self.file_handles.append(f)  # Keep reference to prevent garbage collection
                content = f.read()
                contents.append(content)
            except FileNotFoundError:
                contents.append("")
        return contents
    
    # FIXED: Proper resource management
    def read_files_fixed(self, filenames: List[str]) -> List[str]:
        """FIXED: Files are properly closed using context managers."""
        contents = []
        for filename in filenames:
            try:
                # FIX: Use context manager to ensure file is closed
                with open(filename, 'r') as f:
                    content = f.read()
                    contents.append(content)
            except FileNotFoundError:
                contents.append("")
        return contents
    
    # VULNERABLE: Thread leak
    def start_threads_vulnerable(self, num_threads: int):
        """VULNERABLE: Threads are started but never joined."""
        for i in range(num_threads):
            thread = threading.Thread(target=self._worker_function, args=(i,))
            thread.start()
            self.threads.append(thread)
    
    # FIXED: Proper thread management
    def start_threads_fixed(self, num_threads: int):
        """FIXED: Threads are properly joined."""
        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=self._worker_function, args=(i,))
            thread.start()
            threads.append(thread)
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
    
    def _worker_function(self, thread_id: int):
        """Worker function for threads."""
        time.sleep(0.1)  # Simulate work
        print(f"Thread {thread_id} completed")

# BUG 3: Inefficient Data Structures
class InefficientDataStructures:
    """Demonstrates inefficient use of data structures."""
    
    # VULNERABLE: Using list for membership testing
    def check_membership_vulnerable(self, item: Any, items: List[Any]) -> bool:
        """VULNERABLE: O(n) membership testing with list."""
        return item in items  # O(n) operation
    
    # FIXED: Using set for membership testing
    def check_membership_fixed(self, item: Any, items: List[Any]) -> bool:
        """FIXED: O(1) membership testing with set."""
        items_set = set(items)  # Convert to set once
        return item in items_set  # O(1) operation
    
    # VULNERABLE: Inefficient string concatenation
    def build_string_vulnerable(self, parts: List[str]) -> str:
        """VULNERABLE: String concatenation in loop creates new objects each time."""
        result = ""
        for part in parts:
            result += part  # Creates new string object each iteration
        return result
    
    # FIXED: Efficient string building
    def build_string_fixed(self, parts: List[str]) -> str:
        """FIXED: Using join() for efficient string concatenation."""
        return "".join(parts)  # Single operation, much more efficient

# BUG 4: Unbounded Memory Growth
class UnboundedMemoryExample:
    """Demonstrates unbounded memory growth."""
    
    def __init__(self):
        self.cache = {}
        self.max_cache_size = 1000
    
    # VULNERABLE: Unbounded cache growth
    def get_data_vulnerable(self, key: str) -> str:
        """VULNERABLE: Cache grows indefinitely."""
        if key in self.cache:
            return self.cache[key]
        
        # Simulate expensive computation
        data = f"data_for_{key}"
        self.cache[key] = data  # BUG: No limit on cache size
        return data
    
    # FIXED: Bounded cache with LRU eviction
    def get_data_fixed(self, key: str) -> str:
        """FIXED: Cache has bounded size with LRU eviction."""
        if key in self.cache:
            # Move to end (most recently used)
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        
        # Simulate expensive computation
        data = f"data_for_{key}"
        
        # FIX: Implement LRU eviction when cache is full
        if len(self.cache) >= self.max_cache_size:
            # Remove least recently used item (first item)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = data
        return data

def demonstrate_performance_issues():
    """Demonstrate the performance issues and their fixes."""
    print("=== Performance Issues Demonstration ===\n")
    
    # Test 1: Inefficient Sorting
    print("1. Inefficient Algorithm (Bubble Sort vs Built-in Sort):")
    sorter = InefficientSorter()
    
    # Create a large list for testing
    large_list = list(range(1000, 0, -1))  # 1000 elements in reverse order
    
    # Test bubble sort
    start_time = time.time()
    bubble_result = sorter.bubble_sort_vulnerable(large_list)
    bubble_time = time.time() - start_time
    
    # Test built-in sort
    start_time = time.time()
    builtin_result = sorter.sort_fixed(large_list)
    builtin_time = time.time() - start_time
    
    print(f"   Bubble sort time: {bubble_time:.4f} seconds")
    print(f"   Built-in sort time: {builtin_time:.4f} seconds")
    print(f"   Speed improvement: {bubble_time/builtin_time:.1f}x faster\n")
    
    # Test 2: Memory Leaks
    print("2. Memory Leak (File Handles):")
    memory_example = MemoryLeakExample()
    
    # Create some test files
    test_files = ["test1.txt", "test2.txt", "test3.txt"]
    for filename in test_files:
        with open(filename, 'w') as f:
            f.write(f"Test content for {filename}")
    
    # Test vulnerable method
    print("   Testing vulnerable method (files not closed)...")
    contents = memory_example.read_files_vulnerable(test_files)
    print(f"   Read {len(contents)} files")
    print(f"   Open file handles: {len(memory_example.file_handles)}")
    
    # Test fixed method
    print("   Testing fixed method (files properly closed)...")
    contents = memory_example.read_files_fixed(test_files)
    print(f"   Read {len(contents)} files")
    print(f"   Open file handles: {len(memory_example.file_handles)}")
    
    # Clean up test files
    for filename in test_files:
        try:
            os.remove(filename)
        except:
            pass
    
    # Test 3: Inefficient Data Structures
    print("\n3. Inefficient Data Structures:")
    data_example = InefficientDataStructures()
    
    # Test membership checking
    large_list = list(range(10000))
    test_item = 5000
    
    start_time = time.time()
    for _ in range(1000):
        data_example.check_membership_vulnerable(test_item, large_list)
    list_time = time.time() - start_time
    
    start_time = time.time()
    for _ in range(1000):
        data_example.check_membership_fixed(test_item, large_list)
    set_time = time.time() - start_time
    
    print(f"   List membership test (1000x): {list_time:.4f} seconds")
    print(f"   Set membership test (1000x): {set_time:.4f} seconds")
    print(f"   Speed improvement: {list_time/set_time:.1f}x faster")
    
    # Test string building
    parts = [f"part_{i}" for i in range(1000)]
    
    start_time = time.time()
    result = data_example.build_string_vulnerable(parts)
    concat_time = time.time() - start_time
    
    start_time = time.time()
    result = data_example.build_string_fixed(parts)
    join_time = time.time() - start_time
    
    print(f"   String concatenation: {concat_time:.4f} seconds")
    print(f"   String join: {join_time:.4f} seconds")
    print(f"   Speed improvement: {concat_time/join_time:.1f}x faster")
    
    # Test 4: Unbounded Memory Growth
    print("\n4. Unbounded Memory Growth:")
    memory_growth = UnboundedMemoryExample()
    
    print("   Testing vulnerable method (unbounded cache)...")
    for i in range(2000):
        memory_growth.get_data_vulnerable(f"key_{i}")
    
    print(f"   Cache size after 2000 operations: {len(memory_growth.cache)}")
    
    # Reset for fixed method
    memory_growth.cache.clear()
    
    print("   Testing fixed method (bounded cache)...")
    for i in range(2000):
        memory_growth.get_data_fixed(f"key_{i}")
    
    print(f"   Cache size after 2000 operations: {len(memory_growth.cache)}")
    print(f"   Maximum cache size: {memory_growth.max_cache_size}")

if __name__ == "__main__":
    import os
    demonstrate_performance_issues()