"""
Vercel serverless function entry point for Flask application.
This file wraps the Flask app to work with Vercel's serverless function format.
"""
import sys
import os

# Add the src directory to the Python path so we can import app
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import the Flask app
from app import app

# Vercel's @vercel/python builder automatically detects Flask apps
# Just export the app and Vercel will handle the WSGI wrapping
__all__ = ['app']

