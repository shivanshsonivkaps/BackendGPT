from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
import os
from utils import *
from pine import *
from assistant import *

app = Flask(__name__)
CORS(app)
load_dotenv()

assistant = create_assistant()
asst_id = assistant.id

def checkAuth(auth):
     if not auth:
         raise ValueError('Username and password are required')
     username = auth.username
     password = auth.password
     if username == os.environ.get("USER") and password == os.environ.get("PASSWORD"):
          print("authenticated")
     else:
         raise ValueError('Username and password are invalid')

@app.route("/", methods=["GET"])
def welcome():
    return "Hello, welcome to the API :)"

@app.route("/chat", methods=["POST"])
def chat():
    try:
         checkAuth(request.authorization)
         user_input = request.json.get("question")
         thread = request.json.get("thread")
         if user_input is None or user_input == "":
            return jsonify({'status': 400, 'message': 'Invalid question'}), 400
         similar_data = similar_docs(user_input)
         dataset = similar_data.get("content")
         source = similar_data.get("source")
         update_assistant(dataset, asst_id)
         response = generate_response(user_input, thread, asst_id)
         answer = response + f"\n\nSource : {source}"
         data = {"question": user_input, "answer": answer}
         return jsonify(data)
    except Exception as e:
            print(e)
            error_dict = {"status": "500", "message": str(e)}
            return jsonify(error_dict), 500
   
       

@app.route('/receive_pdf', methods=["POST"])
def receive_pdf():
    try:
        checkAuth(request.authorization)
        files = request.files.getlist('Demo')
        for file in files:
            file_name = file.filename
            newPdf = create_docs(file, file_name)
            push_to_pinecone(newPdf)
        return jsonify("OK"), 200
    except Exception as e:
        print(e)
        error_dict = {"status": "500", "message": str(e)}
        return jsonify(error_dict), 500

if __name__ == '__main__':
    port = int(os.getenv("FLASK_RUN_PORT", 5000))
    app.run(host='0.0.0.0', port=port)
