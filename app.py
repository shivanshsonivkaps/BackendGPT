from flask import Flask,  jsonify, request
from flask_cors import CORS
from utils import *
from pine import *
from langchain_core.messages import AIMessage, HumanMessage


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'



chat_history = []

# Route: /chat
@app.route("/chat", methods=["GET", "POST"])
def qa():
    # Description: Handles a POST request containing a JSON object with a "question" field. It processes the    question, and returns a JSON response containing the question and the answer along with the source of the  information.
    if request.method == "POST":
        try:
            user_input = request.json.get("question")
            sessionID = request.json.get("session")
            if not (sessionID):
               sessionID = generate_unique_key()
               chat_history=[]
    
            if user_input is None and user_input == "":
                newJson = {
                    "status":400,
                    "message":"Invalid question"
                }
                return jsonify(newJson),400
            vector_store = get_vectorstore()
            retriever_chain = get_context_retriever_chain(vector_store)
            conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
            chat_history.append(AIMessage(content=user_input))
            response = conversation_rag_chain.invoke({
                "chat_history": chat_history,
                "input": user_input
            })
            source = similar_docs(user_input)
    
            response = response['answer'] + f"\n\nSource : {source}"

            data = {"Chats":
                        {
                            sessionID : {"answer": response,
                            "question": user_input}
                        }
                    }
            

            return jsonify(data)
        except Exception as e:

            newJson = {
                    "status":400,
                    "message":e
                }
            return jsonify(newJson),400




# Route: /receive_pdf
@app.route('/receive_pdf', methods=["GET",'POST'])
def receive_pdf():
    # Description: Handles a POST request containing a PDF file and pushes it to a Pinecone index for later retrieval.
    try:
        files = request.files
        files = files.getlist('Demo')
        for file in files:
            file_name = file.filename
            print(file_name)
            newPdf = create_docs(file, file_name)
            push_to_pinecone(newPdf)
        return jsonify("OK"), 200
    except Exception as e:
        print(e)
        error_dict = {"status" : "500", "message" : str(e)}
        return jsonify(error_dict), 400

app.run(debug=True, port=5001)
