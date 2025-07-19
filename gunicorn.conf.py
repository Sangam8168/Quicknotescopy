# Gunicorn configuration file
import multiprocessing

# Server socket
bind = "0.0.0.0:${PORT:-10000}"

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "sync"
worker_connections = 1000

# Logging
accesslog = "-"  # Log to stdout
errorlog = "-"   # Log to stderr
loglevel = "info"

# Timeouts
timeout = 120
keepalive = 5
