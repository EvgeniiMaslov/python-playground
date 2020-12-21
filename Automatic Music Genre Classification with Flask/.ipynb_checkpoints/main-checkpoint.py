from flask import Flask, Blueprint, render_template

index = Blueprint('hello', __name__)

@index.route('/')
def hello_world():
    return 'Hello, World!'


if __name__ == '__main__':
    app = Flask(__name__)
    app.register_blueprint(index, url_prefix='/')

    app.run()