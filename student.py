import streamlit as st
import pandas as pd
import joblib
import numpy as np
import uuid
import hashlib
import io
import os
import logging
from io import BytesIO
import matplotlib.pyplot as plt
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


# Function to apply local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load local CSS file
local_css("stylee.css")

# Function to apply page-specific CSS styles
def apply_page_style(page_class):
    st.markdown(f"""
        <style>
        .{page_class} {{
            background-color: inherit;
            color: inherit;
        }}
        .css-1oeqjc1 {{
            background-color: inherit;
            color: inherit;
        }}
        .css-17eq0hr {{
            background-color: inherit;
            color: inherit;
        }}
        </style>
        <div class='{page_class}'></div>
    """, unsafe_allow_html=True)

# Function to hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Function to load users from CSV
def load_users():
    try:
        users_df = pd.read_csv('users.csv')
    except FileNotFoundError:
        users_df = pd.DataFrame(columns=['username', 'password', 'role'])
        users_df.to_csv('users.csv', index=False)

    if 'password' not in users_df.columns:
        users_df['password'] = ''
    if 'role' not in users_df.columns:
        users_df['role'] = None

    users = {}
    for _, row in users_df.iterrows():
        password = row.get('password', '')
        role = row.get('role', None)
        users[row['username']] = {'password': password, 'role': role}

    return users

def apply_page_style(style_class):
    st.markdown(
        f"""
        <style>
        .{style_class} {{
            /* Add your custom styles here */
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Load users and their hashed passwords
users=load_users()

# Load the trained model
model = joblib.load('student_performance_model.pkl')

# Load the dataset for generating distribution
data = pd.read_csv('student_data.csv')

# Preprocessing
data['Gender'] = data['Gender'].map({'M': 0, 'F': 1})
data['Extracurricular_Activities'] = data['Extracurricular_Activities'].map({'No': 0, 'Yes': 1})
data['Previous_Grade'] = data['Previous_Grade'].map({'A': 4, 'B': 3, 'C': 2, 'D': 1, 'F': 0})

# Features for distribution prediction
X = data[['Age', 'Gender', 'Attendance', 'Previous_Grade', 'Study_Hours', 'Extracurricular_Activities']]

# Store login status
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Store page navigation state
if 'page' not in st.session_state:
    st.session_state.page = 'Welcome'

# Store user role
if 'role' not in st.session_state:
    st.session_state.role = None

# Store student data
if 'students' not in st.session_state:
    st.session_state.students = []

# Function for login
def login(username, password):
    if username in users and hash_password(password) == users[username]['password']:
        st.session_state.logged_in = True
        st.session_state.username = username
        st.session_state.role = users[username]['role']
        st.session_state.page = 'Prediction'
        st.success("Login successful!")
    else:
        st.error("Invalid username or password")

# Function for logout
def logout():
    st.session_state.logged_in = False
    st.session_state.page = 'Welcome'
    st.session_state.role = None
    st.success("Logged out successfully!")

# Helper functions to change page state
def set_page_welcome():
    st.session_state.page = 'Welcome'

def set_page_login():
    st.session_state.page = 'Login'

def set_page_register():
    st.session_state.page = 'Register'

def set_page_prediction():
    st.session_state.page = 'Prediction'

def set_page_admin_prediction():
    st.session_state.page = 'Admin Prediction'

def set_page_add_student():
    st.session_state.page = 'Add Student'

def set_page_view_students():
    st.session_state.page = 'View Students'

# Function to render the welcome page
def render_welcome():
    apply_page_style("page_welcome")
    st.title("Welcome to the Student Performance Predictor")
    st.image("result2.jpg", use_column_width=True)
    st.write("Use the navigation menu to log in and access the prediction system.")

# Function to render the login page
# Function to render the login page
# Function to apply page style
def render_login():
    apply_page_style("page-login")
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_button = st.button("Login", key="login_button")  # Add unique key here
    if login_button:
        login(username, password)


# Function to render the registration page
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    users_df = pd.read_csv('users.csv')
    users = {row['username']: {'password': row['password'], 'role': row['role']} for index, row in users_df.iterrows()}
    return users

def save_user(username, password, role):
    users_df = pd.read_csv('users.csv')
    new_user = pd.DataFrame({'username': [username], 'password': [hash_password(password)], 'role': [role]})
    users_df = pd.concat([users_df, new_user], ignore_index=True)
    users_df.to_csv('users.csv', index=False)

def apply_page_style(style_class):
    st.markdown(
        f"""
        <style>
        .{style_class} {{
            /* Add your custom styles here */
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Load users and their hashed passwords
users = load_users()

def render_registration():
    global users  # Global declaration at the top
    apply_page_style("page-registration")
    st.title("Register")
    username = st.text_input("New Username")
    password = st.text_input("New Password", type="password")
    role = st.selectbox("Role", ["admin", "student"])

    # Register button with unique key
    if st.button("Register", key="register_button"):
        if username in users:
            st.error("Username already exists")
        else:
            save_user(username, password, role)
            # Reload the users dictionary after saving a new user
            users = load_users()
            st.success("Registration successful! You can now log in.")
            st.session_state.page = 'Login'
# Function to save student details to CSV file
def save_student_details(student_id, name, age, gender, attendance, previous_grade, study_hours, extracurricular, marks, semester_points):
    student_data = {
        "Student ID": [student_id],
        "Name": [name],
        "Age": [age],
        "Gender": [gender],
        "Attendance": [attendance],
        "Previous Grade": [previous_grade],
        "Study Hours": [study_hours],
        "Extracurricular Activities": [extracurricular],
        "Marks": [marks],
        "Semester Points": [semester_points]
    }
    df = pd.DataFrame(student_data)
    df.to_csv('student_details.csv', mode='a', index=False, header=not os.path.exists('student_details.csv'))
# Function to render the prediction page for students
def render_student_prediction():
    apply_page_style("page-prediction")
    st.title('Student Performance Predictor')

    # Input fields
    name = st.text_input('Student Name')
    age = st.number_input('Age', min_value=10, max_value=100)
    gender = st.selectbox('Gender', ['M', 'F'])
    attendance = st.slider('Attendance (%)', min_value=0, max_value=100)
    previous_grade = st.selectbox('Previous Grade', ['A', 'B', 'C', 'D', 'F'])
    study_hours = st.slider('Study Hours per Week', min_value=0, max_value=40)
    extracurricular = st.selectbox('Extracurricular Activities', ['Yes', 'No'])

    # Map inputs to numerical values
    gender = 0 if gender == 'M' else 1
    extracurricular = 1 if extracurricular == 'Yes' else 0
    previous_grade = {'A': 4, 'B': 3, 'C': 2, 'D': 1, 'F': 0}[previous_grade]

    # Display bar graph for input values
    fig, ax = plt.subplots(figsize=(8, 6))
    labels_bar = ['Attendance (%)', 'Previous Grade', 'Study Hours per Week', 'Extracurricular Activities']
    values_bar = [attendance, previous_grade, study_hours, extracurricular]
    ax.bar(labels_bar, values_bar, color=['blue', 'green', 'purple', 'pink'], alpha=0.7)
    ax.set_xlabel('Factors')
    ax.set_ylabel('Values')
    ax.set_title('Input Values')
    st.pyplot(fig)

    # Predict button
    if st.button('Predict Performance'):
        input_data = np.array([[age, gender, attendance, previous_grade, study_hours, extracurricular]])
        prediction = model.predict(input_data)[0]
        performance = {4: 'A', 3: 'B', 2: 'C', 1: 'D', 0: 'F'}[prediction]
        st.write(f'The predicted performance for {name} is: {performance}')
        
        # Display relevant image based on performance
        if performance == 'A':
            st.image("A.jpg", use_column_width=True)
        elif performance == 'B':
            st.image("B.png", use_column_width=True)
        elif performance == 'C':
            st.image("C.png", use_column_width=True)
        elif performance == 'D':
            st.image("D.jpeg", use_column_width=True)
        else:
            st.image("F.jpg", use_column_width=True)

        # Show prediction distribution (bar graph)
        y_pred = model.predict(X)
        fig2, ax2 = plt.subplots()
        ax2.hist(y_pred, bins=5, edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Performance')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Prediction Distribution (Histogram)')
        st.pyplot(fig2)

        # Display pie chart for prediction results
        grade_counts = pd.Series(y_pred).value_counts().sort_index()
        st.write(grade_counts)  # Debugging step to print grade counts
        fig3, ax3 = plt.subplots()
        labels_pie = grade_counts.index.map({4: 'A', 3: 'B', 2: 'C', 1: 'D', 0: 'F'})
        colors_pie = plt.cm.Paired(range(len(grade_counts)))
        ax3.pie(grade_counts, labels=labels_pie, autopct='%1.1f%%', startangle=90, colors=colors_pie)
        ax3.axis('equal')
        ax3.set_title('Prediction Results (Pie Chart)')
        st.pyplot(fig3)

# Debugging statements like st.write and checking for errors can help in identifying where the issue lies.


# Function to render the add student page for admin

# Function to render the add student details page
# Example usage in your Streamlit function
def render_add_student():
    apply_page_style("page-add-student")
    st.title("Add Student Details")
    
    # Input fields
    student_id = st.text_input('Student ID')
    name = st.text_input('Student Name')
    age = st.number_input('Age', min_value=10, max_value=100)
    gender = st.selectbox('Gender', ['M', 'F'])
    attendance = st.slider('Attendance (%)', min_value=0, max_value=100)
    previous_grade = st.selectbox('Previous Grade', ['A', 'B', 'C', 'D', 'F'])
    study_hours = st.slider('Study Hours per Week', min_value=0, max_value=40)
    extracurricular = st.selectbox('Extracurricular Activities', ['Yes', 'No'])
    
    # Additional fields
    st.write("Enter subject-wise marks:")
    subjects = ['Math', 'Science', 'English', 'History', 'Geography']
    marks = {}
    for subject in subjects:
        marks[subject] = st.number_input(f'{subject} Marks', min_value=0, max_value=100)
    
    st.write("Enter points for all 4 semesters:")
    semesters = ['Semester 1', 'Semester 2', 'Semester 3', 'Semester 4']
    semester_points = {}
    for semester in semesters:
        semester_points[semester] = st.number_input(f'{semester} Points', min_value=0, max_value=10)
    
    # Save button with unique key
    if st.button('Save Details', key="save_student_button"):  # Unique key added here
        save_student_details(student_id, name, age, gender, attendance, previous_grade, study_hours, extracurricular, marks, semester_points)
        st.success('Student details saved successfully!')

# Function to render the view students page for admin
# Function to render the view students page
def render_view_students():
    apply_page_style("page-view-students")
    st.title("View Student Details")

    # Load student details from CSV
    try:
        student_details_df = pd.read_csv('student_details.csv')
    except FileNotFoundError:
        st.warning("No student details found.")
        return

    # Input field to search by student ID
    student_id_input = st.text_input("Enter Student ID to search")

    if student_id_input:
        filtered_df = student_details_df[student_details_df['Student ID'] == student_id_input]
        if filtered_df.empty:
            st.warning("No student found with the given ID.")
            return
    else:
        filtered_df = student_details_df

    # Display student details table
    st.write(filtered_df)

    # Display performance graph for each student
    for index, row in filtered_df.iterrows():
        student_name = row['Name']
        student_id = row['Student ID']

        # Example performance data
        performance_data = np.random.randint(0, 100, size=5)

        # Display performance graph
        fig, ax = plt.subplots()
        ax.plot(['Math', 'Science', 'English', 'History', 'Geography'], performance_data, marker='o')
        ax.set_xlabel('Subjects')
        ax.set_ylabel('Marks')
        ax.set_title(f'Performance Graph for {student_name}')
        st.pyplot(fig)

        # Save performance graph to BytesIO object
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)

        # Button to download performance graph
        st.download_button(
            label=f"Download Performance Graph for {student_name}",
            data=buf,
            file_name=f"{student_name}_performance_graph.png",
            mime="image/png"
        )

        # Generate marksheet as CSV
        marksheet_data = {
            "Subjects": ['Math', 'Science', 'English', 'History', 'Geography'],
            "Marks": performance_data
        }
        marksheet_df = pd.DataFrame(marksheet_data)
        marks_buf = io.BytesIO()
        marksheet_df.to_csv(marks_buf, index=False)
        marks_buf.seek(0)

        # Button to download marks sheet
        st.download_button(
            label=f"Download Marks Sheet for {student_name}",
            data=marks_buf,
            file_name=f"{student_name}_marks_sheet.csv",
            mime="text/csv"
        )


# Example usage: Adding a student
def render_add_student():
    apply_page_style("page-add-student")
    st.title("Add Student Details")
    
    # Input fields
    student_id = st.text_input('Student ID')
    name = st.text_input('Student Name')
    age = st.number_input('Age', min_value=10, max_value=100)
    gender = st.selectbox('Gender', ['M', 'F'])
    attendance = st.slider('Attendance (%)', min_value=0, max_value=100)
    previous_grade = st.selectbox('Previous Grade', ['A', 'B', 'C', 'D', 'F'])
    study_hours = st.slider('Study Hours per Week', min_value=0, max_value=40)
    extracurricular = st.selectbox('Extracurricular Activities', ['Yes', 'No'])
    
    # Additional fields
    st.write("Enter subject-wise marks:")
    subjects = ['Math', 'Science', 'English', 'History', 'Geography']
    marks = {}
    for subject in subjects:
        marks[subject] = st.number_input(f'{subject} Marks', min_value=0, max_value=100)
    
    st.write("Enter points for all 4 semesters:")
    semesters = ['Semester 1', 'Semester 2', 'Semester 3', 'Semester 4']
    semester_points = {}
    for semester in semesters:
        semester_points[semester] = st.number_input(f'{semester} Points', min_value=0, max_value=10)
    
    # Save button with unique key
    if st.button('Save Details', key="save_student_button"):  # Unique key added here
        save_student_details(student_id, name, age, gender, attendance, previous_grade, study_hours, extracurricular, marks, semester_points)
        st.success('Student details saved successfully!')
# Function to apply page-specific CSS styles

# Function to render the About Us page
def render_about_us():
    apply_page_style("page-about-us")
    st.title("About Us")
    st.write("""
        Welcome to the Student Performance Predictor. Our mission is to provide insights into student performance through advanced data analytics and machine learning models. 
        Our team is dedicated to helping students, parents, and educators understand and improve academic outcomes.
    """)
 #``11`````````````   st.image("/home/rguktongole/student_performance/about_us.jpg", use_column_width=True)

# Function to render the Contact Us page

# Function to send email

# Load environment variables from .env file
#load_dotenv()

# Setup logging
logging.basicConfig(filename='app.log', level=logging.DEBUG, format='%(asctime)s %(message)s')

EMAIL = 'nagasrikarangula@gmail.com'
PASSWORD = 'yigv bzhm hwzp jyyn'  # Update with the correct password

def send_email(sender_email, query):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL
        msg['To'] = 'nagasrikarangula@gmail.com'
        msg['Subject'] = 'Query from User'

        body = f"Query from {sender_email}:\n\n{query}"
        msg.attach(MIMEText(body, 'plain'))

        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(EMAIL, PASSWORD)
            server.send_message(msg)

        return True
    except Exception as e:
        print(e)
        return False

def render_contact_us():
    apply_page_style("page-contact-us")
    st.title("Contact Us")
    st.write("""
        If you have any questions or need further assistance, please reach out to us through the following channels:
    """)
    st.write("Email: nagasrikarangula@gmail.com")
    st.write("Phone: +91 98661 15978")
    st.write("Address: Rajiv Gandhi University of Knowledge and Technologies, ONGOLE City, Edu 523001")

    user_email = st.text_input("Your Email")
    user_query = st.text_area("Your Query")

    if st.button("Submit Query"):
        if send_email(user_email, user_query):
            st.success("Your query has been submitted successfully!")
        else:
            st.error("There was an error submitting your query. Please try again later.")

# Ensure your sidebar navigation calls render_contact_us when the Contact Us button is clicked.





# Sidebar navigation
st.sidebar.title("Navigation")
if st.session_state.logged_in:
    st.sidebar.write(f"Logged in as {st.session_state.username} ({st.session_state.role})")
    if st.sidebar.button("Logout"):
        logout()

if st.sidebar.button("Welcome"):
    set_page_welcome()
if st.sidebar.button("Login"):
    set_page_login()
if st.sidebar.button("Register"):
    set_page_register()
if st.sidebar.button("About Us"):
    st.session_state.page = 'About Us'
if st.sidebar.button("Contact Us"):
    st.session_state.page = 'Contact Us'
if st.session_state.logged_in and st.session_state.role == 'student':
    if st.sidebar.button("Prediction"):
        set_page_prediction()
    if st.sidebar.button("View Students"):
        set_page_view_students()
if st.session_state.logged_in and st.session_state.role == 'admin':
    if st.sidebar.button("Prediction"):
        set_page_admin_prediction()
    if st.sidebar.button("Add Student"):
        set_page_add_student()
    if st.sidebar.button("View Students"):
        set_page_view_students()

# Render the appropriate page based on user interaction
if st.session_state.page == 'Welcome':
    render_welcome()
elif st.session_state.page == 'Login':
    render_login()
elif st.session_state.page == 'Register':
    render_registration()
elif st.session_state.page == 'Prediction':
    if st.session_state.role == 'student':
        render_student_prediction()
    elif st.session_state.role == 'admin':
        render_add_student()
elif st.session_state.page == 'Admin Prediction':
    render_student_prediction()
elif st.session_state.page == 'Add Student':
    render_add_student()
elif st.session_state.page == 'View Students':
    render_view_students()
elif st.session_state.page == 'About Us':
    render_about_us()
elif st.session_state.page == 'Contact Us':
    render_contact_us()
