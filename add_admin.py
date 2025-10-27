from pymongo import MongoClient
from werkzeug.security import generate_password_hash
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# MongoDB Atlas connection (replace 'clustername' with your actual cluster name)
MONGO_URI = "mongodb+srv://mydeen_user:Mydeen%40123@clustername.vsl32ko.mongodb.net/mydeen_db?retryWrites=true&w=majority"

def add_admin(username, password):
    client = MongoClient(MONGO_URI)
    db = client.mydeen_db
    if db.admins.find_one({"username": username}):
        logger.warning(f"Admin registration failed: Username {username} already exists")
        print(f"Error: Username {username} already exists")
        client.close()
        return
    db.admins.insert_one({
        "username": username,
        "password": generate_password_hash(password),
        "created_at": datetime.datetime.utcnow()
    })
    logger.info(f"Admin registered: username={username}")
    print(f"Admin {username} added successfully")
    client.close()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python add_admin.py <username> <password>")
        sys.exit(1)
    add_admin(sys.argv[1], sys.argv[2])