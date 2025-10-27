from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from bson import ObjectId
import os
import uuid
import datetime
import logging
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)
# Replace 'clustername' with your actual MongoDB Atlas cluster name
app.config["MONGO_URI"] = "mongodb+srv://mydeen_user:Mydeen%40123@clustername.vsl32ko.mongodb.net/mydeen_db?retryWrites=true&w=majority"
app.config["SECRET_KEY"] = "your-secret-key"
app.config["UPLOAD_FOLDER"] = "static/uploads"
mongo = PyMongo(app)

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Ensure upload folder exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Simulated AI analysis function (placeholder for video analysis)
def analyze_video(video_path, test_type, sit_ups_count=None, vertical_jump_height=None, shuttle_run_time=None):
    logger.debug(f"Analyzing video: {video_path}, test_type: {test_type}, sit_ups_count: {sit_ups_count}, vertical_jump_height: {vertical_jump_height}, shuttle_run_time: {shuttle_run_time}")
    if test_type == "vertical_jump":
        try:
            if vertical_jump_height is None:
                logger.error("Vertical jump height not provided")
                return {"error": "Vertical jump height not provided", "is_valid": False, "cheat_detected": True}
            height = int(vertical_jump_height)
            if height < 0 or height > 100:  # Basic cheat detection
                logger.warning(f"Invalid vertical jump height: {height}")
                return {"height_cm": height, "is_valid": False, "cheat_detected": True}
            return {"height_cm": height, "is_valid": True, "cheat_detected": False}
        except ValueError:
            logger.error("Invalid vertical jump height value")
            return {"height_cm": 0, "is_valid": False, "cheat_detected": True}
    elif test_type == "sit_ups":
        # Auto-detect count from video using MediaPipe Pose
        try:
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5)
            cap = cv2.VideoCapture(video_path)
            count = 0
            direction = 0  # 0: down (lying), 1: up (sitting)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)
                if results.pose_landmarks:
                    lm = results.pose_landmarks.landmark
                    # Coordinates
                    shoulder = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    hip = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x, lm[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    nose = [lm[mp_pose.PoseLandmark.NOSE.value].x, lm[mp_pose.PoseLandmark.NOSE.value].y]
                    # Calculate angle at hip: shoulder-hip-nose
                    def calculate_angle(a, b, c):
                        a = np.array(a)
                        b = np.array(b)
                        c = np.array(c)
                        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
                        angle = np.abs(radians * 180.0 / np.pi)
                        if angle > 180.0:
                            angle = 360 - angle
                        return angle
                    angle = calculate_angle(shoulder, hip, nose)
                    # Thresholds: adjust based on testing (lying ~180°, sitting ~90° or less)
                    if angle < 100 and direction == 0:  # Transition to up
                        count += 1
                        direction = 1
                    elif angle > 150:  # Down position
                        direction = 0
            cap.release()
            if count < 0 or count > 100:  # Basic cheat detection
                logger.warning(f"Invalid sit-up count: {count}")
                return {"count": count, "is_valid": False, "cheat_detected": True}
            return {"count": count, "is_valid": True, "cheat_detected": False}
        except Exception as e:
            logger.error(f"Error analyzing sit-ups video: {e}")
            return {"count": 0, "is_valid": False, "cheat_detected": True}
    elif test_type == "shuttle_run":
        try:
            if shuttle_run_time is None:
                logger.error("Shuttle run time not provided")
                return {"error": "Shuttle run time not provided", "is_valid": False, "cheat_detected": True}
            time_secs = float(shuttle_run_time)
            if time_secs < 5 or time_secs > 60:  # Basic cheat detection
                logger.warning(f"Invalid shuttle run time: {time_secs}")
                return {"time_secs": time_secs, "is_valid": False, "cheat_detected": True}
            return {"time_secs": time_secs, "is_valid": True, "cheat_detected": False}
        except ValueError:
            logger.error("Invalid shuttle run time value")
            return {"time_secs": 0, "is_valid": False, "cheat_detected": True}
    return {"error": "Invalid test type"}

# Simulated performance benchmarking
def benchmark_performance(test_type, value, age, gender):
    benchmarks = {
        "vertical_jump": {"male": {"18-25": 45}, "female": {"18-25": 35}},
        "sit_ups": {"male": {"18-25": 25}, "female": {"18-25": 20}},
        "shuttle_run": {"male": {"18-25": 14}, "female": {"18-25": 15}}
    }
    try:
        age_range = "18-25"  # Simplified for demo; expand for more ranges
        threshold = benchmarks[test_type][gender][age_range]
        # For shuttle run, lower time is better
        meets_standard = value >= threshold if test_type != "shuttle_run" else value <= threshold
        return {"score": value, "meets_standard": meets_standard}
    except:
        return {"score": value, "meets_standard": False}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        data = request.form
        username = data["username"]
        password = generate_password_hash(data["password"])
        age = data["age"]
        gender = data["gender"]
        if mongo.db.users.find_one({"username": username}):
            return jsonify({"error": "Username already exists"}), 400
        mongo.db.users.insert_one({
            "username": username,
            "password": password,
            "age": age,
            "gender": gender,
            "badges": [],
            "created_at": datetime.datetime.utcnow()
        })
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        data = request.form
        user = mongo.db.users.find_one({"username": data["username"]})
        if user and check_password_hash(user["password"], data["password"]):
            session["username"] = data["username"]
            return redirect(url_for("dashboard"))
        return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")

@app.route("/dashboard")
def dashboard():
    if "username" not in session:
        return redirect(url_for("login"))
    user = mongo.db.users.find_one({"username": session["username"]})
    submissions = list(mongo.db.submissions.find({"username": session["username"]}))
    leaderboard = list(mongo.db.submissions.find().sort("score", -1).limit(10))
    return render_template("dashboard.html", user=user, submissions=submissions, leaderboard=leaderboard)

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if "username" not in session:
        return redirect(url_for("login"))
    if request.method == "POST":
        test_type = request.form["test_type"]
        video = request.files["video"]
        vertical_jump_height = request.form.get("vertical_jump_height")
        shuttle_run_time = request.form.get("shuttle_run_time")
        # Sit-ups count is always auto-detected from video, no manual input
        sit_ups_count = None
        if video:
            filename = secure_filename(f"{uuid.uuid4()}.mp4")
            video_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            video.save(video_path)
            # Simulate AI analysis (now auto for sit-ups)
            analysis = analyze_video(video_path, test_type, sit_ups_count, vertical_jump_height, shuttle_run_time)
            if "error" in analysis:
                return jsonify({"error": analysis["error"]}), 400
            user = mongo.db.users.find_one({"username": session["username"]})
            score_value = analysis.get("height_cm", analysis.get("count", analysis.get("time_secs", 0)))
            benchmark = benchmark_performance(test_type, score_value, user["age"], user["gender"])
            submission = {
                "username": session["username"],
                "test_type": test_type,
                "video_path": filename,
                "analysis": analysis,
                "benchmark": benchmark,
                "submitted_at": datetime.datetime.utcnow(),
                "score": score_value
            }
            mongo.db.submissions.insert_one(submission)
            # Award badge if meets standard
            if benchmark["meets_standard"]:
                mongo.db.users.update_one(
                    {"username": session["username"]},
                    {"$addToSet": {"badges": f"{test_type}_achieved"}}
                )
            return redirect(url_for("dashboard"))
    return render_template("upload.html")

@app.route("/edit_submission/<submission_id>", methods=["GET", "POST"])
def edit_submission(submission_id):
    if "username" not in session:
        return redirect(url_for("login"))
    try:
        submission = mongo.db.submissions.find_one({"_id": ObjectId(submission_id)})
    except:
        return jsonify({"error": "Invalid submission ID"}), 400
    if not submission or submission["username"] != session["username"]:
        return jsonify({"error": "Unauthorized"}), 403
    user = mongo.db.users.find_one({"username": session["username"]})
    if request.method == "POST":
        test_type = request.form["test_type"]
        vertical_jump_height = request.form.get("vertical_jump_height")
        shuttle_run_time = request.form.get("shuttle_run_time")
        # Sit-ups count is always auto-detected
        sit_ups_count = None
        new_video_path = submission["video_path"]
        new_video = request.files.get("video")
        filename = None
        if new_video and new_video.filename != "":
            # Delete old video file
            old_path = os.path.join(app.config["UPLOAD_FOLDER"], submission["video_path"])
            if os.path.exists(old_path):
                os.remove(old_path)
            # Save new video
            filename = secure_filename(f"{uuid.uuid4()}.mp4")
            new_video_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            new_video.save(new_video_path)
        # Simulate AI analysis with new values (re-analyze video always for sit-ups, manual for others)
        analysis = analyze_video(new_video_path, test_type, sit_ups_count, vertical_jump_height, shuttle_run_time)
        if "error" in analysis:
            return render_template("edit_submission.html", submission=submission, user=user, error=analysis["error"])
        score_value = analysis.get("height_cm", analysis.get("count", analysis.get("time_secs", 0)))
        benchmark = benchmark_performance(test_type, score_value, user["age"], user["gender"])
        # Prepare update data
        update_data = {
            "test_type": test_type,
            "analysis": analysis,
            "benchmark": benchmark,
            "score": score_value,
            "submitted_at": datetime.datetime.utcnow()
        }
        if filename:
            update_data["video_path"] = filename
        # Update submission
        mongo.db.submissions.update_one(
            {"_id": ObjectId(submission_id)},
            {"$set": update_data}
        )
        # Handle badges: remove old if it met standard, add new if meets
        old_test_type = submission["test_type"]
        old_badge = f"{old_test_type}_achieved"
        new_badge = f"{test_type}_achieved"
        if submission["benchmark"]["meets_standard"]:
            # Check if this was the last submission for old test type that met standard
            other_submissions = mongo.db.submissions.find({
                "username": session["username"],
                "test_type": old_test_type,
                "benchmark.meets_standard": True
            }).count()
            if other_submissions <= 1:  # This was the only one, remove badge
                mongo.db.users.update_one(
                    {"username": session["username"]},
                    {"$pull": {"badges": old_badge}}
                )
        if benchmark["meets_standard"]:
            mongo.db.users.update_one(
                {"username": session["username"]},
                {"$addToSet": {"badges": new_badge}}
            )
        return redirect(url_for("dashboard"))
    # For GET, prefill form values from current submission
    prefill = {}
    if submission["test_type"] == "vertical_jump":
        prefill["vertical_jump_height"] = submission["analysis"].get("height_cm", "")
    elif submission["test_type"] == "shuttle_run":
        prefill["shuttle_run_time"] = submission["analysis"].get("time_secs", "")
    # For sit-ups, no prefill needed as auto-detected
    return render_template("edit_submission.html", submission=submission, user=user, prefill=prefill)

@app.route("/delete_submission/<submission_id>", methods=["POST"])
def delete_submission(submission_id):
    if "username" not in session:
        return redirect(url_for("login"))
    try:
        submission = mongo.db.submissions.find_one({"_id": ObjectId(submission_id)})
    except:
        return jsonify({"error": "Invalid submission ID"}), 400
    if not submission or submission["username"] != session["username"]:
        return jsonify({"error": "Unauthorized"}), 403
    # Delete video file
    video_path = os.path.join(app.config["UPLOAD_FOLDER"], submission["video_path"])
    if os.path.exists(video_path):
        os.remove(video_path)
    # Delete submission
    mongo.db.submissions.delete_one({"_id": ObjectId(submission_id)})
    # Optionally remove badge if no other qualifying submissions for this test type
    test_type = submission["test_type"]
    remaining = list(mongo.db.submissions.find({
        "username": session["username"],
        "test_type": test_type,
        "benchmark.meets_standard": True
    }))
    if not remaining:
        mongo.db.users.update_one(
            {"username": session["username"]},
            {"$pull": {"badges": f"{test_type}_achieved"}}
        )
    return redirect(url_for("dashboard"))

@app.route("/admin", methods=["GET", "POST"])
def admin():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        logger.debug(f"Admin login attempt: username={username}")
        admin = mongo.db.admins.find_one({"username": username})
        if admin and check_password_hash(admin["password"], password):
            session["admin"] = True
            logger.info("Admin login successful")
            return redirect(url_for("admin_dashboard"))
        logger.warning("Admin login failed: Invalid credentials")
        return render_template("admin_login.html", error="Invalid admin credentials")
    return render_template("admin_login.html")

@app.route("/admin/register", methods=["GET", "POST"])
def admin_register():
    # Restrict to existing admins (or allow first admin creation if none exist)
    if mongo.db.admins.count_documents({}) > 0 and "admin" not in session:
        logger.warning("Unauthorized admin registration attempt")
        return jsonify({"error": "Unauthorized"}), 403
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        logger.debug(f"Admin registration attempt: username={username}")
        if mongo.db.admins.find_one({"username": username}):
            logger.warning("Admin registration failed: Username exists")
            return render_template("admin_register.html", error="Username already exists")
        mongo.db.admins.insert_one({
            "username": username,
            "password": generate_password_hash(password),
            "created_at": datetime.datetime.utcnow()
        })
        logger.info(f"Admin registered: username={username}")
        return redirect(url_for("admin"))
    return render_template("admin_register.html")

@app.route("/admin/dashboard")
def admin_dashboard():
    if "admin" not in session:
        logger.warning("Unauthorized access to admin dashboard")
        return redirect(url_for("admin"))
    submissions = list(mongo.db.submissions.find())
    return render_template("admin_dashboard.html", submissions=submissions)

@app.route("/logout")
def logout():
    session.pop("username", None)
    session.pop("admin", None)
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)