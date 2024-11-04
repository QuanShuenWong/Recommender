from flask import Flask, request, render_template
from predictor import get_recommendations

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_id = request.form["user_id"]
        recommendations = get_recommendations(user_id)
        return render_template("results.html", recommendations=recommendations)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)