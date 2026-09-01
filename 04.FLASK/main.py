from flask import Flask

# Create Flask app
app = Flask(__name__)

# Route
@app.route("/")
def welcome():
    return "Welcome to the jungle 🚀"

# Run server
if __name__ == "__main__":
    app.run(debug=True)