import streamlit as st

st.session_state.chapters = st.session_state.main_variable

# Initialize the list of dictionaries
# if 'chapters' not in st.session_state:
#     st.session_state.chapters = [
#         {
#             'chapter': 'Chapter 1: The Electoral Landscape',
#             'points': [
#                 'Introduction to Indian elections',
#                 'History and evolution of the Indian electoral system',
#                 'Key features and principles of Indian elections',
#                 'Role of the Election Commission of India',
#                 'Electoral constituencies and reservation system'
#             ]
#         },
#         {
#             'chapter': 'Chapter 2: Political Parties and Alliances',
#             'points': [
#                 'Overview of the Indian party system',
#                 'Major national and regional political parties',
#                 'Ideological orientations and platforms of parties',
#                 'Coalitions and alliances in Indian elections',
#                 'Impact of party system on electoral outcomes'
#             ]
#         }
#     ]

def add_chapter(chapter, points):
    st.session_state.chapters.append({'chapter': chapter, 'points': points})
    st.success(f"Chapter '{chapter}' has been added.")

def update_chapter(index, chapter, points):
    st.session_state.chapters[index] = {'chapter': chapter, 'points': points}
    st.success(f"Chapter '{chapter}' has been updated.")

def delete_chapter(index):
    chapter = st.session_state.chapters.pop(index)
    st.success(f"Chapter '{chapter['chapter']}' has been deleted.")

st.title("Chapters and Sub topics Editor")

# Display the current list of chapters
st.subheader("Current Chapters and Sub topics")
for idx, chapter_info in enumerate(st.session_state.chapters):
    with st.expander(f"{chapter_info['chapter']}"):
        st.write(f"Points: {chapter_info['points']}")

# Form to add a new chapter
st.subheader("Add New Chapter")
with st.form(key='add_chapter_form'):
    new_chapter = st.text_input('Chapter Title')
    new_points = st.text_area('Sub topics (separate each point with a semicolon)')
    add_chapter_button = st.form_submit_button(label='Add Chapter')

    if add_chapter_button and new_chapter and new_points:
        points_list = [point.strip() for point in new_points.split(';')]
        add_chapter(new_chapter, points_list)

# Form to update an existing chapter
st.subheader("Update Existing Chapter")
with st.form(key='update_chapter_form'):
    chapter_index = st.number_input('Chapter Index to Update', min_value=0, max_value=len(st.session_state.chapters)-1, step=1)
    updated_chapter = st.text_input('Updated Chapter Title')
    updated_points = st.text_area('Updated Points (separate each point with a semicolon)')
    update_chapter_button = st.form_submit_button(label='Update Chapter')

    if update_chapter_button and updated_chapter and updated_points:
        updated_points_list = [point.strip() for point in updated_points.split(';')]
        update_chapter(chapter_index, updated_chapter, updated_points_list)

# Form to delete a chapter
st.subheader("Delete Chapter")
with st.form(key='delete_chapter_form'):
    chapter_index_to_delete = st.number_input('Chapter Index to Delete', min_value=0, max_value=len(st.session_state.chapters)-1, step=1)
    delete_chapter_button = st.form_submit_button(label='Delete Chapter')

    if delete_chapter_button:
        delete_chapter(chapter_index_to_delete)

# Display the updated list of chapters
st.subheader("Updated Chapters and Sub topics")
for idx, chapter_info in enumerate(st.session_state.chapters):
    with st.expander(f"{chapter_info['chapter']}"):
        st.write(f"Points: {chapter_info['points']}")


if st.button("Generate E Book"):
    st.session_state.main_variable = st.session_state.chapters
    st.switch_page("pages/generate.py")