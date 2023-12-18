from flask import Flask, render_template
import threading
import time

app = Flask(__name__)

def background_task():
   
    for i in range(5):
        print(f"Task in progress... Step {i + 1}")
        time.sleep(2)
    print("Task completed.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_task')
def start_task():
    thread = threading.Thread(target=background_task)
    thread.start()
    return "Background task started. Check the console for updates."

if __name__ == '__main__':
    app.run(debug=True)
