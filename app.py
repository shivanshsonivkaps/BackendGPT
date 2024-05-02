from flask import Flask, jsonify, request
from flask_cors import CORS
from utils import *
from pine import *
from assistant import *

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
assistant = create_assistant()
asst_id = assistant.id

@app.route("/chat", methods=["POST"])
def qa():
        try:
            user_input = request.json.get("question") 
            thread = request.json.get("thread")  
            
            if user_input is None or user_input == "":
                newJson = {
                    "status": 400,
                    "message": "Invalid question"
                }
                return jsonify(newJson), 400

            similar_data = similar_docs(user_input)
            dataset = similar_data.get("content")
            source = similar_data.get("source")
            update_assistant(dataset,asst_id)
            response = generate_response(user_input, thread,  asst_id)
            answer = response + f"\n\nSource : {source}"
            data = {"question": user_input, "answer": answer }
            return jsonify(data)
        except Exception as e:
         print(e)
         error_dict = {"status": "500", "message": str(e)}
         return jsonify(error_dict), 400

@app.route('/receive_pdf', methods=["POST"])
def receive_pdf():
    try:
        files = request.files
        files = files.getlist('Demo')
        for file in files:
            file_name = file.filename
            newPdf = create_docs(file, file_name)
            push_to_pinecone(newPdf)
        return jsonify("OK"), 200
    except Exception as e:
        print(e)
        error_dict = {"status": "500", "message": str(e)}
        return jsonify(error_dict), 400


if __name__ == '__main__':
    # Get port from environment variable, fallback to default 5000 if not set
    port = int(os.getenv("FLASK_RUN_PORT", 5000))
    app.run(host='0.0.0.0', port=port)