import streamlit as st
import joblib

# 1. Load the saved model and vectorizer
@st.cache_resource # Keeps the model in memory so it doesn't reload on every click
def load_assets():
    model = joblib.load("spam_model.pkl")
    cv = joblib.load("vector.pkl")
    return model, cv

try:
    model, cv = load_assets()
except Exception as e:
    st.error(f"Error loading model or vectorizer: {e}")
    st.stop()

# 2. Set up the Dashboard UI
st.set_page_config(page_title="SMS Spam Detector", page_icon="💬", layout="centered")

st.title("💬 SMS Spam Detection Dashboard")
st.write("Type an SMS message below to check if it's **Ham** (safe) or **Spam**.")

st.divider()

# 3. User Input
user_input = st.text_area("Enter SMS text here:", height=150, placeholder="Type your message...")

# 4. Prediction Logic
if st.button("Analyze Message", type="primary"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        # Transform the input text using the loaded CountVectorizer
        msg_vector = cv.transform([user_input])
        
        # Predict using the loaded MultinomialNB model
        prediction = model.predict(msg_vector)[0]
        
        # Handle predictions (Assuming 1 or 'spam' for spam, 0 or 'ham' for ham)
        # Looking at your notebook, model.predict returned [1] for ham, but let's make it flexible:
        st.subheader("Result:")
        
        if prediction in [1, 'spam', 'Spam']:
            st.error("🚨 **Warning: This looks like SPAM!**")
        else:
            st.success("✅ **Safe: This looks like HAM.**")

st.divider()
st.caption("Built with Streamlit • Machine Learning Model: Multinomial Naive Bayes")