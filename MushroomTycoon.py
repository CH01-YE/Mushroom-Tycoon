

#Contents of ~/my_app/streamlit_app.py
import streamlit as st

# Contents of ~/my_app/streamlit_app.py
import streamlit as st
import base64

st.markdown('<h1 style="color:black;">Mushroom Tycoon</h1>', unsafe_allow_html=True)
st.markdown('<h2 style="color:black;">Predict and judge the condtion of mushroom through its growth environment data and images.</h2>', unsafe_allow_html=True)
# st.markdown('<h3 style="color:gray;"> made by Jiyu Won, Yeeun Choi</h3>', unsafe_allow_html=True)

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file) 
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: scroll; # doesn't work
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg('../streamlit/mushroom.webp')

# def main_page():
#     st.markdown("# Main page 🎈")
#     st.sidebar.markdown("# Main page 🎈")

# def page2():
#     st.markdown("# Page 2 ❄️")
#     st.sidebar.markdown("# Page 2 ❄️")

# def page3():
#     st.markdown("# Page 3 🎉")
#     st.sidebar.markdown("# Page 3 🎉")

# page_names_to_funcs = {
#     "Main Page": main_page,
#     "Page 2": page2,
#     "Page 3": page3,
# }

# selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
# page_names_to_funcs[selected_page]()