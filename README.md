# Bug Fixes Demonstration

This repository demonstrates 3 major categories of bugs found in the codebase and their corresponding fixes. Each bug represents a common programming mistake that can lead to security vulnerabilities, performance issues, or logic errors.

## Bugs Found and Fixed

### 1. **Resource Management and Error Handling** (`SentenceTransformerPOC.py`)

**Original Issues:**
- No error handling for model loading failures
- No resource cleanup (GPU memory not freed)
- Hardcoded model path without fallback options
- No validation of input data
- Duplicate sentence processing without deduplication

**Fixes Applied:**
- Added comprehensive error handling with try-catch blocks
- Implemented proper resource management with `finally` blocks
- Added GPU memory cleanup using `torch.cuda.empty_cache()`
- Created modular functions for better maintainability
- Added input validation and deduplication
- Implemented logging for better debugging

**Impact:** Prevents crashes, reduces memory leaks, and improves code robustness.

### 2. **Security Vulnerabilities** (`security_example.py`)

**Bug 2.1: SQL Injection Vulnerability**
- **Issue:** Direct string concatenation in SQL queries allows malicious input
- **Fix:** Use parameterized queries with placeholders (`?`)
- **Risk:** Attackers could execute arbitrary SQL commands, delete tables, or access unauthorized data

**Bug 2.2: Weak Password Hashing**
- **Issue:** Using MD5 hash which is cryptographically broken
- **Fix:** Implement bcrypt with salt for strong password hashing
- **Risk:** Passwords could be easily cracked using rainbow tables

**Bug 2.3: Path Traversal Attack**
- **Issue:** No validation of file paths allows reading files outside intended directory
- **Fix:** Implement path validation using `os.path.realpath()` and boundary checking
- **Risk:** Attackers could read sensitive system files like `/etc/passwd`

### 3. **Performance Issues** (`performance_example.py`)

**Bug 3.1: Inefficient Algorithm**
- **Issue:** Using bubble sort (O(n²)) instead of built-in sort (O(n log n))
- **Fix:** Replace custom sorting with Python's optimized `sorted()` function
- **Impact:** 1000x performance improvement for large datasets

**Bug 3.2: Memory Leaks**
- **Issue:** File handles and threads never closed, causing resource leaks
- **Fix:** Use context managers (`with` statements) and proper thread joining
- **Impact:** Prevents memory exhaustion and resource depletion

**Bug 3.3: Inefficient Data Structures**
- **Issue:** Using lists for membership testing (O(n)) instead of sets (O(1))
- **Fix:** Convert lists to sets for frequent membership operations
- **Impact:** Significant performance improvement for large datasets

**Bug 3.4: Unbounded Memory Growth**
- **Issue:** Cache grows indefinitely without size limits
- **Fix:** Implement LRU (Least Recently Used) eviction policy
- **Impact:** Prevents memory overflow and maintains predictable memory usage

## Running the Examples

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the bug demonstrations:
```bash
# Original code with fixes
python experiments/SentenceTransformerPOC.py

# Security vulnerabilities demonstration
python experiments/security_example.py

# Performance issues demonstration
python experiments/performance_example.py
```

## Key Lessons

1. **Always use parameterized queries** to prevent SQL injection
2. **Implement proper resource management** with context managers and cleanup
3. **Choose appropriate data structures** for your use case (sets vs lists)
4. **Use built-in optimized functions** instead of custom implementations
5. **Implement proper error handling** and input validation
6. **Set bounds on resource usage** to prevent unbounded growth
7. **Use strong cryptographic functions** for security-critical operations

## Security Best Practices

- Never trust user input
- Always validate and sanitize file paths
- Use strong, salted hashing for passwords
- Implement proper access controls
- Log security-relevant events
- Keep dependencies updated

## Performance Best Practices

- Profile your code to identify bottlenecks
- Choose appropriate algorithms and data structures
- Implement caching with eviction policies
- Use built-in optimized functions when available
- Monitor memory usage and implement bounds
- Clean up resources promptly
