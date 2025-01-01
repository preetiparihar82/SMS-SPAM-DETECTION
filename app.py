import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK stopwords if not already installed
nltk.download('stopwords')

# Load the pre-trained model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))  # Load your trained model here
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))  # Load your vectorizer here

# Function to preprocess the SMS text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    # Join words back to string
    return " ".join(words)

# Streamlit App UI
st.set_page_config(page_title="SMS Spam Detection", layout="wide")

# Sidebar for Information
st.sidebar.title("Spam Email Classifier")
st.sidebar.markdown("""
    **Welcome to the Email Spam Detection App!**  
    This app classifies Email messages as **Spam** or **Ham (Not Spam)** using machine learning.

    ### How it Works:
    1. **Input your SMS message** in the provided text area.
    2. Click on the **Submit** button to classify the Email.
    3. The result will show whether the Email is **Spam** or **Ham**.

    ### About the Model:
    The machine learning model was trained on Email data and uses **TF-IDF Vectorization** to convert the message into a numerical format for classification.

   

    Developer: Preeti Parihar                              
""")

# Main content area
st.title("Spam Detection")
st.markdown("""
    This app uses machine learning to classify SMS and Email messages as either **Spam** or **Ham** (Not Spam).  
    The machine learning model was trained on a dataset of SMS messages and classifies messages based on their content.
""")

# Display some example messages (optional)
st.markdown("### Example SMS Messages:")
st.write("1. Free money now!!!")
st.write("2. Hi, how are you?")
st.write("3. Congrats, you've won a lottery")



sms_input = st.text_area("Enter Text:", "")
submit_button = st.button("Submit")

if submit_button:
    if sms_input:
        # Preprocess the input message
        cleaned_sms = preprocess_text(sms_input)
        
        # Vectorize the cleaned message using the pre-trained vectorizer
        vectorized_sms = vectorizer.transform([cleaned_sms])
        
        # Make prediction using the pre-trained model
        prediction = model.predict(vectorized_sms)
        
        # Display the result
        if prediction == 1:
            st.warning("This message is **Spam**.")
        else:
            st.success("This message is **Not Spam (Ham)**.")
    else:
        st.error("Please enter an SMS message to classify.")


