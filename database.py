import sqlite3
import os
from datetime import datetime
from typing import Optional

# Database file path
DB_PATH = os.path.join(os.path.dirname(__file__), "tour_guide.db")


def init_database():
    """Initialize the SQLite database and create tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create user_preferences table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_preferences (
            user_id TEXT PRIMARY KEY,
            raw_preferences TEXT NOT NULL,
            enhanced_preferences TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_PATH}")


def get_db_connection():
    """Get a database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Return rows as dictionaries
    return conn


def create_user_preferences(user_id: str, raw_preferences: str, enhanced_preferences: str) -> bool:
    """
    Create a new user preference entry.

    Args:
        user_id: Unique user identifier
        raw_preferences: User's original input
        enhanced_preferences: AI-enhanced version

    Returns:
        True if successful, False otherwise
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO user_preferences (user_id, raw_preferences, enhanced_preferences)
            VALUES (?, ?, ?)
        """, (user_id, raw_preferences, enhanced_preferences))

        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        # User ID already exists
        return False
    except Exception as e:
        print(f"Error creating user preferences: {e}")
        return False


def get_user_preferences(user_id: str) -> Optional[dict]:
    """
    Get user preferences by user_id.

    Args:
        user_id: Unique user identifier

    Returns:
        Dictionary with preference data or None if not found
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT user_id, raw_preferences, enhanced_preferences, created_at, updated_at
            FROM user_preferences
            WHERE user_id = ?
        """, (user_id,))

        row = cursor.fetchone()
        conn.close()

        if row:
            return dict(row)
        return None
    except Exception as e:
        print(f"Error getting user preferences: {e}")
        return None


def update_user_preferences(user_id: str, raw_preferences: str, enhanced_preferences: str) -> bool:
    """
    Update existing user preferences.

    Args:
        user_id: Unique user identifier
        raw_preferences: User's original input
        enhanced_preferences: AI-enhanced version

    Returns:
        True if successful, False otherwise
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE user_preferences
            SET raw_preferences = ?,
                enhanced_preferences = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE user_id = ?
        """, (raw_preferences, enhanced_preferences, user_id))

        rows_affected = cursor.rowcount
        conn.commit()
        conn.close()

        return rows_affected > 0
    except Exception as e:
        print(f"Error updating user preferences: {e}")
        return False


def delete_user_preferences(user_id: str) -> bool:
    """
    Delete user preferences.

    Args:
        user_id: Unique user identifier

    Returns:
        True if successful, False otherwise
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            DELETE FROM user_preferences
            WHERE user_id = ?
        """, (user_id,))

        rows_affected = cursor.rowcount
        conn.commit()
        conn.close()

        return rows_affected > 0
    except Exception as e:
        print(f"Error deleting user preferences: {e}")
        return False


# Initialize database on module import
init_database()
