from flask import Flask, render_template, jsonify, request
from flask_cors import CORS, cross_origin
from utils import *
from pine import *

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


# home()
# Route: /
# Description: Returns a simple "Hello World" message.
@app.route("/")
def home():
    return "Hello World"


# qa()
# Route: /chat
# Description: Handles a POST request containing a JSON object with a "question" field. It processes the question, retrieves relevant information using retrieval chains, and returns a JSON response containing the question and the answer along with the source of the information.
@app.route("/chat", methods=["GET", "POST"])
def qa():
    if request.method == "POST":
        try:
            user_input = request.json.get("question")
            print(user_input)
            if user_input is None and user_input == "":
                newJson = {
                    "status":400,
                    "message":"Invalid question"
                }
                return jsonify(newJson),400
            vector_store = get_vectorstore()
            retriever_chain = get_context_retriever_chain(vector_store)
            conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
            chat_history = []
            response = conversation_rag_chain.invoke({
                "chat_history": chat_history,
                "input": user_input
            })
            source = similar_docs(user_input)
            response = response['answer'] + f"\n\nSource : {source}"

            data = {"question": user_input, "answer": response}

            return jsonify(data)
        except Exception as e:

            newJson = {
                    "status":400,
                    "message":e
                }
            return jsonify(newJson),400
        

# receive_pdf()
# Route: /receive_pdf
# Description: Handles a POST request containing a PDF file named "Demo". It extracts text from the PDF, processes it, and pushes it to a Pinecone index for later retrieval.
@app.route('/receive_pdf', methods=["GET",'POST'])
def receive_pdf():
    try:
    #Getting files from API
        files = request.files
        file = files.get('Demo')
        file_name = file.filename
        newPdf = create_docs(file , file_name)

        push_to_pinecone(newPdf)
        return jsonify("OK"),200
    except Exception as e:
        print(e)
        return({
            "status" : "500",
            "message" : "error"
        }),400

app.run(debug=True, port=5001)