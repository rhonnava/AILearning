import sqlite3
import hashlib
import os
from typing import Optional

# BUG 1: SQL Injection Vulnerability
class UserManager:
    def __init__(self, db_path: str = "users.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with user table."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                email TEXT
            )
        ''')
        conn.commit()
        conn.close()
    
    # VULNERABLE: SQL Injection
    def get_user_by_username_vulnerable(self, username: str) -> Optional[dict]:
        """VULNERABLE: This method is susceptible to SQL injection attacks."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # BUG: Direct string concatenation allows SQL injection
        query = f"SELECT * FROM users WHERE username = '{username}'"
        cursor.execute(query)
        
        user = cursor.fetchone()
        conn.close()
        
        if user:
            return {
                'id': user[0],
                'username': user[1],
                'password_hash': user[2],
                'email': user[3]
            }
        return None
    
    # FIXED: Parameterized Query
    def get_user_by_username_safe(self, username: str) -> Optional[dict]:
        """SAFE: This method uses parameterized queries to prevent SQL injection."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # FIX: Use parameterized query with placeholders
        query = "SELECT * FROM users WHERE username = ?"
        cursor.execute(query, (username,))
        
        user = cursor.fetchone()
        conn.close()
        
        if user:
            return {
                'id': user[0],
                'username': user[1],
                'password_hash': user[2],
                'email': user[3]
            }
        return None
    
    def add_user(self, username: str, password: str, email: str = None) -> bool:
        """Add a new user with hashed password."""
        try:
            # Hash the password before storing
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Use parameterized query for safety
            cursor.execute(
                "INSERT INTO users (username, password_hash, email) VALUES (?, ?, ?)",
                (username, password_hash, email)
            )
            
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            return False  # Username already exists
        except Exception as e:
            print(f"Error adding user: {e}")
            return False

# BUG 2: Weak Password Hashing
class WeakPasswordManager:
    def __init__(self):
        self.users = {}
    
    # VULNERABLE: Weak password hashing
    def hash_password_weak(self, password: str) -> str:
        """VULNERABLE: Using MD5 which is cryptographically broken."""
        import hashlib
        return hashlib.md5(password.encode()).hexdigest()
    
    # FIXED: Strong password hashing with salt
    def hash_password_strong(self, password: str) -> str:
        """SAFE: Using bcrypt with salt for strong password hashing."""
        import bcrypt
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def verify_password_strong(self, password: str, hashed: str) -> bool:
        """Verify password against bcrypt hash."""
        import bcrypt
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

# BUG 3: Insecure File Operations
class FileManager:
    def __init__(self, base_dir: str = "/tmp"):
        self.base_dir = base_dir
    
    # VULNERABLE: Path traversal attack
    def read_file_vulnerable(self, filename: str) -> str:
        """VULNERABLE: Allows path traversal attacks."""
        file_path = os.path.join(self.base_dir, filename)
        
        # BUG: No path validation - allows reading files outside base_dir
        with open(file_path, 'r') as f:
            return f.read()
    
    # FIXED: Secure file operations
    def read_file_safe(self, filename: str) -> str:
        """SAFE: Prevents path traversal attacks."""
        # FIX: Validate and sanitize the file path
        file_path = os.path.join(self.base_dir, filename)
        real_path = os.path.realpath(file_path)
        
        # Ensure the resolved path is within the base directory
        if not real_path.startswith(os.path.realpath(self.base_dir)):
            raise ValueError("Access denied: Path traversal detected")
        
        # Additional security: check if file exists and is readable
        if not os.path.isfile(real_path):
            raise FileNotFoundError("File not found")
        
        with open(real_path, 'r') as f:
            return f.read()

def demonstrate_vulnerabilities():
    """Demonstrate the security vulnerabilities and their fixes."""
    print("=== Security Vulnerabilities Demonstration ===\n")
    
    # Test SQL Injection
    print("1. SQL Injection Vulnerability:")
    user_manager = UserManager()
    
    # Add a test user
    user_manager.add_user("alice", "password123", "alice@example.com")
    
    # Normal usage
    user = user_manager.get_user_by_username_safe("alice")
    print(f"   Safe query result: {user['username'] if user else 'Not found'}")
    
    # Malicious input that could cause SQL injection
    malicious_input = "'; DROP TABLE users; --"
    print(f"   Malicious input: {malicious_input}")
    print("   With vulnerable method, this could delete the entire users table!")
    print("   With safe method, it's treated as a literal string.\n")
    
    # Test weak password hashing
    print("2. Weak Password Hashing:")
    weak_manager = WeakPasswordManager()
    
    password = "mypassword123"
    weak_hash = weak_manager.hash_password_weak(password)
    print(f"   Weak MD5 hash: {weak_hash}")
    print("   MD5 is cryptographically broken and should not be used for passwords!\n")
    
    # Test path traversal
    print("3. Path Traversal Vulnerability:")
    file_manager = FileManager("/tmp")
    
    # Try to read a file outside the intended directory
    malicious_filename = "../../../etc/passwd"
    print(f"   Malicious filename: {malicious_filename}")
    print("   This could allow reading system files outside the intended directory!")
    
    try:
        # This should fail with the safe method
        content = file_manager.read_file_safe(malicious_filename)
        print("   SUCCESS (this shouldn't happen)")
    except ValueError as e:
        print(f"   BLOCKED: {e}")
    except FileNotFoundError as e:
        print(f"   BLOCKED: {e}")

if __name__ == "__main__":
    demonstrate_vulnerabilities()