# Importing required libraries
import logging
import os
import sys
import time

from flask import Flask
from src.routes import api
from dotenv import load_dotenv
from watchdog.observers import Observer
from watchdog.events import LoggingEventHandler

# Load environment variables from .env and .flaskenv files
load_dotenv()

def create_app(test_config=None):
    # Load environment variables from .env and .flaskenv files
    load_dotenv()

    app = Flask(__name__, instance_relative_config=True)

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # Apply blueprints to the app
    app.register_blueprint(api, url_prefix='/api')

    @app.route('/')
    def welcome():
        return "<h1>Welcome to your rag application!</h1>"

    if __name__ == '__main__':
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
        path = sys.argv[1] if len(sys.argv) > 1 else '.'
        event_handler = LoggingEventHandler()
        observer = Observer()
        observer.schedule(event_handler, path, recursive=True)
        observer.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()
        app.run(debug=True)

    return app