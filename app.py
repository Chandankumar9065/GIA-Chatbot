from flask import Flask, render_template, request, redirect, send_file, session, url_for
import json
import random
import pickle
from datetime import datetime
import csv
import os
from collections import Counter

from train import train_model

app = Flask(__name__)
app.secret_key = os.environ.get("GIA_SECRET_KEY", "gia-admin-session-secret")
INTENTS_FILE = "intents.json"
USERS_FILE = "users.csv"
CHAT_LOGS_FILE = "chat_logs.csv"
VISITOR_COUNT_FILE = "visitor_count.txt"
ADMIN_USER = os.environ.get("GIA_ADMIN_USER", "admin")
ADMIN_PASSWORD = os.environ.get("GIA_ADMIN_PASSWORD", "admin123")


def load_intents():
    with open(INTENTS_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def load_model_files():
    trained_model = pickle.load(open("model.pkl", "rb"))
    trained_vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return trained_model, trained_vectorizer


def read_csv_rows(filename):
    if not os.path.isfile(filename):
        return []

    with open(filename, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def append_csv_row(filename, headers, row):
    file_exists = os.path.isfile(filename)

    with open(filename, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=headers)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def increment_visitor_count():
    count = get_visitor_count() + 1

    with open(VISITOR_COUNT_FILE, "w", encoding="utf-8") as file:
        file.write(str(count))

    return count


def get_visitor_count():
    if not os.path.isfile(VISITOR_COUNT_FILE):
        return 0

    with open(VISITOR_COUNT_FILE, "r", encoding="utf-8") as file:
        value = file.read().strip()

    return int(value) if value.isdigit() else 0


def log_chat_query(message, response, prediction="", probability=""):
    now = datetime.now()

    append_csv_row(
        CHAT_LOGS_FILE,
        ["Question", "Response", "Intent", "Confidence", "Date", "Time"],
        {
            "Question": message,
            "Response": response,
            "Intent": prediction,
            "Confidence": probability,
            "Date": now.strftime("%d-%m-%Y"),
            "Time": now.strftime("%H:%M:%S")
        }
    )


def build_analytics(users, logs):
    intent_counts = Counter(row.get("Intent", "unknown") or "unknown" for row in logs)
    date_counts = Counter(row.get("Date", "unknown") or "unknown" for row in logs)

    return {
        "visitor_count": get_visitor_count(),
        "user_count": len(users),
        "query_count": len(logs),
        "intent_count": len(intent_counts),
        "top_intents": intent_counts.most_common(6),
        "daily_queries": sorted(date_counts.items(), reverse=True)[:7]
    }


def add_question_answer(tag, question, answer):
    global data, model, vectorizer

    tag = tag.strip().lower().replace(" ", "_")
    question = question.strip()
    answer = answer.strip()

    if not tag or not question or not answer:
        return False

    data = load_intents()

    for intent in data["intents"]:
        if intent["tag"] == tag:
            if question not in intent["patterns"]:
                intent["patterns"].append(question)
            if answer not in intent["responses"]:
                intent["responses"].append(answer)
            break
    else:
        data["intents"].append({
            "tag": tag,
            "patterns": [question],
            "responses": [answer]
        })

    with open(INTENTS_FILE, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

    model, vectorizer = train_model()
    data = load_intents()
    return True


def admin_is_logged_in():
    return session.get("admin_logged_in") is True


def require_admin():
    if not admin_is_logged_in():
        return redirect(url_for("admin_login", next=request.path))

    return None


data = load_intents()

model, vectorizer = load_model_files()

print("Model Loaded Successfully!")
print("Available Tags:", model.classes_)


@app.route("/")
def home():
    increment_visitor_count()
    return render_template("index.html")


@app.route("/admin")
def admin_panel():
    blocked = require_admin()

    if blocked:
        return blocked

    users = read_csv_rows(USERS_FILE)
    logs = read_csv_rows(CHAT_LOGS_FILE)
    analytics = build_analytics(users, logs)

    return render_template(
        "admin.html",
        users=list(reversed(users))[:100],
        logs=list(reversed(logs))[:150],
        analytics=analytics,
        message=request.args.get("message", "")
    )


@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if admin_is_logged_in():
        return redirect(url_for("admin_panel"))

    error = ""

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if username == ADMIN_USER and password == ADMIN_PASSWORD:
            session["admin_logged_in"] = True
            return redirect(request.args.get("next") or url_for("admin_panel"))

        error = "Invalid user ID or password."

    return render_template("admin_login.html", error=error)


@app.route("/admin/logout")
def admin_logout():
    session.pop("admin_logged_in", None)
    return redirect(url_for("admin_login"))


@app.route("/admin/add_qa", methods=["POST"])
def admin_add_qa():
    blocked = require_admin()

    if blocked:
        return blocked

    tag = request.form.get("tag", "")
    question = request.form.get("question", "")
    answer = request.form.get("answer", "")
    added = add_question_answer(tag, question, answer)
    message = "Q&A added and model retrained." if added else "Please fill all Q&A fields."

    return redirect(url_for("admin_panel", message=message))


@app.route("/admin/download_chat_logs")
def download_chat_logs():
    blocked = require_admin()

    if blocked:
        return blocked

    if not os.path.isfile(CHAT_LOGS_FILE):
        with open(CHAT_LOGS_FILE, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Question", "Response", "Intent", "Confidence", "Date", "Time"])

    return send_file(CHAT_LOGS_FILE, as_attachment=True, download_name="chat_logs.csv")

@app.route("/save_user", methods=["POST"])
def save_user():

    data = request.get_json()

    name = data.get("name")
    mobile = data.get("mobile")

    now = datetime.now()

    append_csv_row(
        USERS_FILE,
        ["Name", "Contact", "Date", "Time"],
        {
            "Name": name,
            "Contact": mobile,
            "Date": now.strftime("%d-%m-%Y"),
            "Time": now.strftime("%H:%M:%S")
        }
    )

    return {"status":"success"}


@app.route("/get")
def chatbot_response():

    msg = request.args.get('msg')

    if not msg:
        return "Please enter a question."

    msg = msg.lower().strip()

    try:

        # Greetings ko direct handle karo
        greetings = [
            "hi", "hii", "hello", "hey",
            "good morning", "good evening",
            "good afternoon"
        ]

        if msg in greetings:
            response = "Hello! Welcome to Galgotias University AI Assistant. How can I help you today?"
            log_chat_query(msg, response, "greeting", "direct")
            return response

        # Vectorize text
        X = vectorizer.transform([msg])

        # Predict
        prediction = model.predict(X)[0]

        # Confidence
        probability = max(model.predict_proba(X)[0])

        print("\n====================")
        print("Message =", msg)
        print("Prediction =", prediction)
        print("Probability =", probability)

        # Strong confidence check
        if probability < 0.05:
            response = "Sorry! I could not understand your question. Please ask about admissions, courses, fees, placements, hostel, scholarship or campus facilities."
            log_chat_query(msg, response, prediction, round(probability, 4))
            return response

        # Find response
        for intent in data["intents"]:

            if intent["tag"] == prediction:

                response = random.choice(intent["responses"])
                log_chat_query(msg, response, prediction, round(probability, 4))
                return response

        response = "Sorry! Response not found."
        log_chat_query(msg, response, prediction, round(probability, 4))
        return response

    except Exception as e:

        print("ERROR =", e)
        response = "Internal Server Error"
        log_chat_query(msg, response, "error", "")
        return response


if __name__ == "__main__":
    app.run(debug=True)
