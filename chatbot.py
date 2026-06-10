import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import scrolledtext

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)


faqs = [
    {
        "question": "How do I track my order?",
        "answer": "Go to 'My Orders' in your account and click 'Track Shipment'. You can also use the tracking number sent to your email."
    },
    {
        "question": "How long does shipping take?",
        "answer": "Standard shipping takes 5-7 business days. Express (2-3 days) and overnight options are available at checkout."
    },
    {
        "question": "Do you ship internationally?",
        "answer": "Yes, we ship to 50+ countries. International orders take 10-15 business days. Customs duties may apply."
    },
    {
        "question": "What is the return policy?",
        "answer": "You can return unused items within 30 days of delivery. Go to 'My Orders' and click 'Return Item' to start the process."
    },
    {
        "question": "My item arrived damaged. What do I do?",
        "answer": "Contact our support team within 48 hours with photos of the damage. We will send a replacement or give you a full refund."
    },
    {
        "question": "How do I get a refund?",
        "answer": "Refunds go back to your original payment method within 5-7 business days after we receive your return."
    },
    {
        "question": "What payment methods do you accept?",
        "answer": "We accept Visa, Mastercard, PayPal, UPI, and net banking. All payments are secured with SSL encryption."
    },
    {
        "question": "How do I apply a promo code?",
        "answer": "Enter your code in the 'Discount Code' box at checkout and click Apply. One code per order."
    },
    {
        "question": "How do I reset my password?",
        "answer": "Click 'Forgot Password' on the login page. We will email you a reset link that is valid for 24 hours."
    },
    {
        "question": "How do I cancel my order?",
        "answer": "You can cancel within 1 hour of placing the order from the 'My Orders' page. After that it goes into processing and cannot be cancelled."
    },
    {
        "question": "How do I contact customer support?",
        "answer": "Email us at support@example.com or call 1-800-123-4567 (Mon-Fri, 9am-6pm). Live chat is also available on our website."
    },
]


# Clean and preprocess text using NLTK

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 1]
    return ' '.join(tokens)



faq_questions_cleaned = [preprocess(f['question']) for f in faqs]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(faq_questions_cleaned)

# Match user input to closest FAQ
def get_answer(user_question):
    cleaned = preprocess(user_question)
    user_vec = vectorizer.transform([cleaned])
    scores = cosine_similarity(user_vec, tfidf_matrix).flatten()
    best_match = scores.argmax()
    best_score = scores[best_match]

    if best_score < 0.2:
        return "Sorry, I don't have an answer for that. Try rephrasing or contact support@example.com"

    return faqs[best_match]['answer']


#  Tkinter UI
def send_message(event=None):
    user_text = entry.get().strip()
    if not user_text:
        return

    chat_box.config(state=tk.NORMAL)
    chat_box.insert(tk.END, f"You:  {user_text}\n", "user")

    answer = get_answer(user_text)
    chat_box.insert(tk.END, f"Bot:  {answer}\n\n", "bot")

    chat_box.config(state=tk.DISABLED)
    chat_box.see(tk.END)
    entry.delete(0, tk.END)


window = tk.Tk()
window.title("FAQ Chatbot")
window.geometry("600x500")
window.resizable(False, False)
window.configure(bg="#f0f0f0")

tk.Label(window, text="FAQ Chatbot", font=("Helvetica", 16, "bold"),
         bg="#f0f0f0", fg="#333").pack(pady=(15, 5))

chat_box = scrolledtext.ScrolledText(window, state=tk.DISABLED,
                                     width=70, height=22,
                                     font=("Helvetica", 11),
                                     bg="#ffffff", fg="#222",
                                     relief=tk.FLAT, padx=10, pady=10)
chat_box.pack(padx=15, pady=5)
chat_box.tag_config("user", foreground="#1a73e8", font=("Helvetica", 11, "bold"))
chat_box.tag_config("bot",  foreground="#333333", font=("Helvetica", 11))

chat_box.config(state=tk.NORMAL)
chat_box.insert(tk.END, "Bot:  Hi!chatbot.py Ask me anything about orders, shipping, returns, or payments.\n\n", "bot")
chat_box.config(state=tk.DISABLED)
input_frame = tk.Frame(window, bg="#f0f0f0")
input_frame.pack(padx=15, pady=(5, 15), fill=tk.X)

entry = tk.Entry(input_frame, font=("Helvetica", 12), relief=tk.FLAT,
                 bg="#ffffff", fg="#222", insertbackground="#333",
                 highlightthickness=1, highlightbackground="#ccc",
                 highlightcolor="#1a73e8")
entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8, padx=(0, 10))
entry.bind("<Return>", send_message)
entry.focus()

send_btn = tk.Button(input_frame, text="Send", font=("Helvetica", 11, "bold"),
                     bg="#1a73e8", fg="white", relief=tk.FLAT,
                     activebackground="#1558b0", activeforeground="white",
                     padx=18, pady=8, cursor="hand2",
                     command=send_message)
send_btn.pack(side=tk.RIGHT)

window.mainloop()
