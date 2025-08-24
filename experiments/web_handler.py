import hashlib
import pickle
import base64
import bcrypt
import secrets
from datetime import datetime, timedelta

class WebHandler:
    def __init__(self):
        self.sessions = {}
        self.login_attempts = {}
    
    def authenticate_user(self, username: str, password: str) -> bool:
        """Authenticate user - now using secure bcrypt hashing"""
        stored_hash = self.get_stored_password_hash(username)
        if stored_hash is None:
            return False
        
        # Use bcrypt for secure password verification
        try:
            return bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8'))
        except ValueError:
            # Handle case where stored hash is not valid bcrypt
            return False
    
    def get_stored_password_hash(self, username: str) -> str:
        """Simulate getting stored password hash"""
        # Mock stored bcrypt hash for demo (hash of "hello")
        return "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj3QJOvHin4a"
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        # Generate a salt and hash the password
        salt = bcrypt.gensalt()
        password_hash = bcrypt.hashpw(password.encode('utf-8'), salt)
        return password_hash.decode('utf-8')
    
    def deserialize_user_data(self, data: str) -> dict:
        """Deserialize user data - pickle vulnerability"""
        # Bug: Using pickle on untrusted data (security vulnerability)
        decoded_data = base64.b64decode(data)
        return pickle.loads(decoded_data)
    
    def track_login_attempts(self, ip_address: str) -> bool:
        """Track login attempts - no rate limiting"""
        current_time = datetime.now()
        
        if ip_address not in self.login_attempts:
            self.login_attempts[ip_address] = []
        
        # Bug: No rate limiting, allowing brute force attacks
        self.login_attempts[ip_address].append(current_time)
        
        # Should implement rate limiting here
        return True
    
    def generate_session_token(self, user_id: str) -> str:
        """Generate session token - weak randomness"""
        # Bug: Predictable session token generation
        timestamp = str(datetime.now().timestamp())
        token = hashlib.md5(f"{user_id}{timestamp}".encode()).hexdigest()
        return token
    
    def validate_input(self, user_input: str) -> str:
        """Validate and sanitize input - XSS vulnerability"""
        # Bug: No XSS protection
        return user_input  # Should sanitize HTML/script tags
    
    def log_error(self, error: Exception, user_data: dict) -> None:
        """Log error - information disclosure"""
        # Bug: Logging sensitive information
        print(f"Error: {error}")
        print(f"User data: {user_data}")  # May contain passwords, tokens, etc.