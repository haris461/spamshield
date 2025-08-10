#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load dataset and train model
df = pd.read_csv("emails.csv")

X = df.drop(columns=['Email No.', 'Prediction'])
y = df['Prediction']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate model accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

# Calculate top spam-indicating words
spam_emails = X[y == 1]
ham_emails = X[y == 0]

spam_word_counts = spam_emails.sum(axis=0)
ham_word_counts = ham_emails.sum(axis=0)

spam_ratio = (spam_word_counts + 1) / (ham_word_counts + 1)  # +1 to avoid zero division
top_spam_words = spam_ratio.sort_values(ascending=False).head(20)

# Function to convert email text to vector
def email_to_vector(email_text, feature_columns):
    vector = np.zeros((1, len(feature_columns)))
    for word in email_text.lower().split():
        if word in feature_columns:
            idx = feature_columns.get_loc(word)
            vector[0, idx] += 1
    return vector

# Page config and styling
st.set_page_config(
    page_title="Spam Email Detector",
    page_icon="📧",
    layout="wide"
)

# CSS styles for button and classification report
st.markdown(
    """
    <style>
    .title {
        font-size: 44px;
        font-weight: 700;
        color: #2E86C1;
        text-align: center;
        margin-bottom: 0.3em;
    }
    .subtitle {
        font-size: 18px;
        color: #555555;
        text-align: center;
        margin-bottom: 2em;
    }
    .footer {
        font-size: 12px;
        color: #999999;
        text-align: center;
        margin-top: 4em;
    }

    /* Yellow button styling */
    div.stButton > button:first-child {
        background-color: #FFD700 !important;  /* gold/yellow */
        color: black !important;
        font-weight: 700;
        border-radius: 8px;
        height: 45px;
        width: 180px;
        transition: background-color 0.3s ease;
        border: none;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
    }
    div.stButton > button:first-child:hover {
        background-color: #FFC300 !important;  /* darker yellow */
        color: black !important;
    }

    /* Classification report table styling */
    .class-report {
        border-collapse: collapse;
        width: 100%;
        margin-top: 1em;
        font-family: monospace;
    }
    .class-report th, .class-report td {
        border: 1px solid #ddd;
        padding: 10px;
        text-align: center;
    }
    .class-report th {
        background-color: #2E86C1;
        color: white;
        font-weight: 700;
    }
    .class-report td.precision {
        background-color: #d4edda;  /* light green */
    }
    .class-report td.recall {
        background-color: #fff3cd;  /* light yellow */
    }
    .class-report td.f1-score {
        background-color: #f8d7da;  /* light red */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<h1 class="title">📧 SpamShield (Spam Email Detector)</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Enter the content of an email below and check if it is spam or not.</p>', unsafe_allow_html=True)

# Sidebar info
with st.sidebar:
    st.header("Instructions & Info")
    st.write(
        f"""
        - Enter the email text in the input box on the main page.  
        - Click **Check if Spam** to get the prediction.  
        - The model is trained on a sample dataset and may have false positives.  
        - Accuracy on test set: **{accuracy * 100:.2f}%**
        """
    )
    st.markdown("---")
    st.subheader("Example Emails")
    example_emails = [
        "Free money offer! Click now to win big prizes!",
        "Please find attached the meeting agenda for tomorrow.",
        "Congratulations, you have won a lottery. Claim your reward immediately.",
        "Can we reschedule the project update call to next Monday?"
    ]
    selected_example = st.selectbox("Choose example to autofill", [""] + example_emails)
    if selected_example:
        st.write("Example email filled in input box.")

# Main input area with autofill from sidebar selection
user_input = st.text_area(
    "Type or paste your email text here:",
    height=160,
    max_chars=1500,
    value=selected_example if selected_example else ""
)

check_button = st.button("Check if Spam")

if check_button:
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text before checking.")
    else:
        vec = email_to_vector(user_input, X.columns)
        prediction = model.predict(vec)[0]
        if prediction == 1:
            st.error("🚫 This email is **Spam**. Be careful!")
        else:
            st.success("✅ This email is **Not Spam**.")

# Colored classification report display using HTML
def colored_classification_report(report_dict):
    html = '<table class="class-report">'
    html += '<tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr>'
    for key, metrics in report_dict.items():
        if key == "accuracy":
            continue
        html += (
            f'<tr>'
            f'<td>{key}</td>'
            f'<td class="precision">{metrics.get("precision", 0):.2f}</td>'
            f'<td class="recall">{metrics.get("recall", 0):.2f}</td>'
            f'<td class="f1-score">{metrics.get("f1-score", 0):.2f}</td>'
            f'<td>{metrics.get("support", 0)}</td>'
            f'</tr>'
        )
    html += '</table>'
    return html

with st.expander("See Model Classification Report"):
    st.markdown("Precision, Recall, F1-Score on Test Data:")
    st.markdown(colored_classification_report(report), unsafe_allow_html=True)

# Show top spam words
st.markdown("---")
st.markdown("### Top 20 Spam-Indicating Words")
with st.container():
    for word, score in top_spam_words.items():
        st.markdown(f"- **{word}**: spam ratio = {score:.2f}")

st.markdown('<p class="footer">Developed by YourName | Powered by Streamlit & Scikit-learn</p>', unsafe_allow_html=True)



# In[ ]:





# In[ ]:





# In[ ]:




