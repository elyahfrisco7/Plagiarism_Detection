# app/__init__.py
import os
from flask import Flask
from app.config import UPLOAD_FOLDER
from app.routes import register_routes

def create_app():
    """Factory function to create and configure the Flask application"""
    app = Flask(__name__, 
                template_folder='../templates',
                static_folder='../static')
    
    # Configuration
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    
    # Create upload folder if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    # Register routes
    register_routes(app)
    
    return app
