import streamlit as st
import sys
import streamlit as st
from docx import Document
from io import BytesIO
import zipfile
import os

def create_word_doc(text,cnt):
    # Create a new Document
    doc = Document()
    
    # Add text to the document
    doc.add_paragraph(text)
    if not os.path.exists('eBook'):
        os.makedirs('eBook')
    doc_path = f'eBook/out{cnt}.docx'
    doc.save(doc_path)
    return doc_path

def zip_folder(folder_path, zip_name):
    with zipfile.ZipFile(zip_name, 'w') as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), folder_path))

# adding Folder_2 to the system path
sys.path.insert(0, 'D:/4th year/WeekendAI/e book/app.py')

from app import generate_text

st.write("hey this is page 1")

output = st.session_state.main_variable

st.write(output)
llm = st.session_state.llm
# output = st.session_state.output
# serp_api_key = st.session_state.serp_api_key
outline_str = st.session_state.outline_str


if st.button("Generate E Book"):
    cnt=1
    for i in output:
        keys = i.keys()
        key_list = list(keys)
        chapter = i[key_list[0]]
        sub_topics = i[key_list[1]]
        generated_content = generate_text(llm, chapter, sub_topics)

        word_file = create_word_doc(generated_content,cnt)
        cnt = cnt + 1
    create_word_doc(outline_str,0)
    zip_name = 'eBook.zip'
    zip_folder('eBook', zip_name)
    
    # Read the zip file and provide it as a download
    with open(zip_name, 'rb') as f:
        bytes_data = f.read()
        st.download_button(
            label='Download eBook Zip',
            data=bytes_data,
            file_name=zip_name,
            mime='application/zip'
        )

    # Clean up the zip file
    os.remove(zip_name)
    
    for file in os.listdir('eBook'):
        os.remove(os.path.join('eBook', file))
    os.rmdir('eBook')
