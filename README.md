
This project demonstrates a simple implementation of a Retrieval Augmented Generation (RAG) system. 

**Key Features:**

* **Built with:** Python and Streamlit
* **Dependencies:**
    * ChromaDB: 0.5.23
    * PDF Plumber: 0.11.4
    * OpenAI: 1.57.0

**Requirements:**

* **OpenAI API Key:** You'll need an active OpenAI API key to use this application.
* **Endpoint:** The endpoint for your OpenAI model (e.g., `text-davinci-003`).

**Setup:**

1. **Install dependencies:**
   ```bash
   pip install chromadb pdfplumber openai streamlit
Create an .env file:

Create a file named .env in the project's root directory.

Add the following lines to the .env file:

OPENAI_API_KEY="your_openai_api_key"
OPENAI_ENDPOINT="your endpoint" 
Replace your_openai_api_key with your actual OpenAI API key.

Prerequisite :
Give the location of your pdfs in the configuration section of the code
Run the application:

Bash

streamlit run app.py 
Usage:


Ask your questions: Enter your questions in the provided input field.
Get answers: The application will use the RAG system to retrieve relevant information from the uploaded PDFs and generate an answer based on your query.
Note:

This is a simple example and may require further customization and optimization for real-world applications.
Please refer to the OpenAI API documentation for more information on available models and their usage.
Contributing:

Contributions are welcome! Please feel free to fork this repository and submit a pull request.