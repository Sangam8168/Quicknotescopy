# Gunicorn configuration file
import multiprocessing
import os

# Server socket - use PORT environment variable if available, otherwise default to 10000
bind = "0.0.0.0:{}".format(os.environ.get("PORT", "10000"))

# Worker processes - reduce workers for Render's free tier
workers = 1  # Reduced from multiprocessing.cpu_count() * 2 + 1
worker_class = "sync"
worker_connections = 1000

# Logging
accesslog = "-"  # Log to stdout
errorlog = "-"   # Log to stderr
loglevel = "info"

# Timeouts - increased for model loading
timeout = 300  # Increased from 120
keepalive = 5
