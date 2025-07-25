// FILE: app.py
from flask import Flask, request, jsonify, render_template
from answer_bot import get_answer

app = Flask(__name__, static_folder="static", template_folder="templates")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("message", "")
    if not question:
        return jsonify({"reply": "No question provided."})

    answer = get_answer(question)
    return jsonify({"reply": answer})

if __name__ == "__main__":
    app.run(debug=True)



// FILE: answer_bot.py
import fitz  # PyMuPDF
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai

openai.api_key = "YOUR_OPENAI_API_KEY"

# Load and chunk PDF
with fitz.open("textbook.pdf") as doc:
    full_text = " ".join([page.get_text() for page in doc])

def chunk_text(text, chunk_size=200):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

chunks = chunk_text(full_text)
vectorizer = TfidfVectorizer().fit(chunks)
chunk_vectors = vectorizer.transform(chunks)

def get_answer(question):
    question_vec = vectorizer.transform([question])
    sims = cosine_similarity(question_vec, chunk_vectors)[0]
    top_indices = sims.argsort()[-3:][::-1]
    context = "\n".join([chunks[i] for i in top_indices])

    prompt = f"""Use the text below to answer the question:

Text:
{context}

Question:
{question}

Answer:"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"



<!-- FILE: templates/index.html -->
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AI Chatbot</title>
  <script defer src="/static/main.js"></script>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="min-h-screen bg-gray-100 p-4 flex flex-col items-center">
  <div class="w-full max-w-2xl h-[90vh] flex flex-col bg-white rounded-2xl shadow-lg">
    <div id="chatContainer" class="flex-1 overflow-y-auto space-y-2 p-4"></div>
    <div class="p-4 border-t flex gap-2">
      <input
        id="userInput"
        type="text"
        placeholder="Type your message..."
        class="flex-1 border rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-400"
        onkeydown="if(event.key === 'Enter') sendMessage()"
      />
      <button
        onclick="sendMessage()"
        class="bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600"
      >Send</button>
    </div>
  </div>
</body>
</html>



// FILE: static/main.js
async function sendMessage() {
  const input = document.getElementById("userInput");
  const chatContainer = document.getElementById("chatContainer");
  const message = input.value.trim();
  if (!message) return;

  const userMsg = document.createElement("div");
  userMsg.className = "flex justify-end";
  userMsg.innerHTML = `<div class='bg-blue-500 text-white rounded-2xl px-4 py-2 max-w-[75%] shadow mb-2'>${message}</div>`;
  chatContainer.appendChild(userMsg);
  chatContainer.scrollTop = chatContainer.scrollHeight;
  input.value = "";

  const loader = document.createElement("div");
  loader.id = "loader";
  loader.className = "flex justify-start";
  loader.innerHTML = `
    <div class='rounded-2xl px-4 py-2 bg-gray-200 text-gray-800 shadow flex items-center mb-2'>
      <svg class='animate-spin mr-2 h-4 w-4 text-gray-600' xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24'>
        <circle class='opacity-25' cx='12' cy='12' r='10' stroke='currentColor' stroke-width='4'></circle>
        <path class='opacity-75' fill='currentColor' d='M4 12a8 8 0 018-8v8z'></path>
      </svg>
      Thinking...
    </div>`;
  chatContainer.appendChild(loader);
  chatContainer.scrollTop = chatContainer.scrollHeight;

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message })
    });
    const data = await res.json();
    document.getElementById("loader").remove();

    const botMsg = document.createElement("div");
    botMsg.className = "flex justify-start";
    botMsg.innerHTML = `<div class='bg-gray-200 text-gray-800 rounded-2xl px-4 py-2 max-w-[75%] shadow mb-2'>${data.reply}</div>`;
    chatContainer.appendChild(botMsg);
    chatContainer.scrollTop = chatContainer.scrollHeight;
  } catch (err) {
    document.getElementById("loader").remove();
    const errorMsg = document.createElement("div");
    errorMsg.className = "flex justify-start";
    errorMsg.innerHTML = `<div class='bg-gray-200 text-gray-800 rounded-2xl px-4 py-2 max-w-[75%] shadow mb-2'>Error fetching response.</div>`;
    chatContainer.appendChild(errorMsg);
    chatContainer.scrollTop = chatContainer.scrollHeight;
  }
}
