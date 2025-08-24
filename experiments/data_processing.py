import json
import os
from typing import List, Dict, Any

class DataProcessor:
    def __init__(self):
        self.cache = {}
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file"""
        with open(config_path, 'r') as file:
            return json.load(file)
    
    def process_user_data(self, user_input: str) -> str:
        """Process user input - now using parameterized queries"""
        # Use parameterized queries to prevent SQL injection
        import re
        
        # Input validation: only allow alphanumeric characters and spaces
        if not re.match(r'^[a-zA-Z0-9\s]+$', user_input):
            raise ValueError("Invalid input: only alphanumeric characters and spaces allowed")
        
        # Use parameterized query (placeholder for actual implementation)
        query = "SELECT * FROM users WHERE name = ?"
        # In real implementation, pass user_input as parameter to database driver
        return query, user_input
    
    def calculate_average(self, numbers: List[float]) -> float:
        """Calculate average - now handles empty lists"""
        if not numbers:
            raise ValueError("Cannot calculate average of empty list")
        
        total = sum(numbers)
        return total / len(numbers)
    
    def find_max_value(self, data: List[int]) -> int:
        """Find maximum value - contains logic error"""
        if not data:
            return None
        
        max_val = data[0]
        for i in range(len(data)):
            if data[i] > max_val:
                max_val = data[i]
        return max_val
    
    def cache_expensive_operation(self, key: str, data: List[Dict]) -> List[Dict]:
        """Cache data - memory leak potential"""
        if key not in self.cache:
            # Memory leak: never clearing cache
            processed_data = []
            for item in data:
                processed_data.append({**item, 'processed': True})
            self.cache[key] = processed_data
        return self.cache[key]
    
    def validate_email(self, email: str) -> bool:
        """Validate email - weak validation"""
        return "@" in email and "." in email
    
    def process_file_batch(self, file_paths: List[str]) -> None:
        """Process multiple files - resource leak"""
        files = []
        for path in file_paths:
            file_handle = open(path, 'r')
            files.append(file_handle)
            # Bug: files never closed, resource leak
        
        # Process files...
        for file_handle in files:
            content = file_handle.read()
            # Process content...