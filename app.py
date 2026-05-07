import streamlit as st
from datetime import date
import pandas as pd
import os
import time

from csv_storage import (
    save_employee,
    update_employee,
    save_embedding,
    EMP_FILE
)

from Register_Camera import capture_faces_streamlit
from Mark_Attendance_Camera import capture_attendance_face_streamlit
from Embedding_Matcher import match_face, database_embeddings, mark_attendance_logic

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Face Attendance System", layout="centered")
st.title("📋 Facial Recognition Attendance System")

# -------------------------------------------------
# MODE SELECTION
# -------------------------------------------------
mode = st.radio(
    "Select Operation",
    ["Register Employee", "Mark Attendance", "View Attendance", "Update Employee Data"],
    horizontal=True
)
st.markdown("---")

# =================================================
# REGISTER EMPLOYEE
# =================================================
if mode == "Register Employee":

    st.subheader("Employee Registration Form")

    col1, col2 = st.columns(2)

    with col1:
        emp_id = st.text_input("Employee ID", placeholder="e.g. E101")
        emp_phone = st.text_input("Phone Number")

    with col2:
        emp_name = st.text_input("Employee Name")
        emp_date = st.date_input("Registration Date", value=date.today())

    emp_email = st.text_input("Employee Email")
    emp_address = st.text_area("Employee Address", height=80)

    st.markdown("---")

    # ---------------------------
    # CAMERA BUTTON
    # ---------------------------
    if st.button("📷 Open Camera"):

        if not emp_id or not emp_name:
            st.warning("Please fill Employee ID and Name first.")
            st.stop()

        st.info("Camera is opening... Please look at the camera.")

        frame_placeholder = st.empty()
        progress_text = st.empty()
        final_mean_embedding = None

        for frame, embeddings, mean_embedding in capture_faces_streamlit(max_faces=10):
            if frame is not None:
                frame_placeholder.image(frame, channels="RGB")
                if embeddings is not None:
                    progress_text.info(f"Capturing Faces: {len(embeddings)} / 10")

            if mean_embedding is not None:
                final_mean_embedding = mean_embedding
                break

        if final_mean_embedding is not None:
            st.success("Face Capture Completed ✅")
            st.session_state["mean_embedding"] = final_mean_embedding
        else:
            st.error("Face capture failed. Please try again.")

    # ---------------------------
    # SAVE EMPLOYEE DATA
    # ---------------------------
    if "mean_embedding" in st.session_state:

        if st.button("💾 Save Employee Data"):

            if not emp_id or not emp_name:
                st.warning("Please fill Employee ID and Name.")
                st.stop()

            # Duplicate ID Check
            if os.path.exists(EMP_FILE):
                df_check = pd.read_csv(EMP_FILE)
                if emp_id in df_check["Employee_ID"].astype(str).values:
                    st.error("Employee ID already exists ❌")
                    st.stop()

            try:
                save_employee(
                    emp_id,
                    emp_name,
                    emp_phone,
                    emp_email,
                    emp_address,
                    str(emp_date)
                )

                save_embedding(
                    emp_id,
                    st.session_state["mean_embedding"]
                )

                st.success("Employee Registered Successfully ✅")
                del st.session_state["mean_embedding"]
                st.rerun()

            except Exception as e:
                st.error(f"Error saving employee: {e}")

# =================================================
# MARK ATTENDANCE (Continuous Mode)
# =================================================
elif mode == "Mark Attendance":

    st.subheader("📸 Mark Attendance")
    st.write("Look at the camera to mark your attendance.")

    # Load employee info for name lookup
    df1 = pd.read_csv("Data/employees_info.csv")

    frame_placeholder = st.empty()
    message_placeholder = st.empty()  # Placeholder to show messages

    # Loop through the camera feed
    for frame, embeddings, keypoints, mean_embedding in capture_attendance_face_streamlit(max_faces=5):

        # Display camera frame
        if frame is not None:
            frame_placeholder.image(frame, channels="RGB")

        # If a face is detected and mean embedding is available
        if mean_embedding is not None:
            emp_id, score = match_face(mean_embedding, database_embeddings)

            if emp_id:
                # Get employee name safely
                mask = df1["Employee_ID"].astype(str) == str(emp_id)
                if mask.any():
                    emp_name = df1.loc[mask, "Employee_Name"].values[0]

                    # Mark attendance using your logic (time & status handled inside)
                    success, msg = mark_attendance_logic(emp_id, emp_name)

                    if success:
                        # Show success message for 2 seconds
                        message_placeholder.success(f"✅ Attendance marked for {emp_name} (ID: {emp_id})")
                        time.sleep(2)  # pause for 2 seconds
                        message_placeholder.empty()
                    else:
                        # If attendance logic returns error (e.g., outside time)
                        message_placeholder.warning(msg)
                        time.sleep(2)
                        message_placeholder.empty()
                else:
                    message_placeholder.error(f"Employee ID {emp_id} not found")
                    time.sleep(2)
                    message_placeholder.empty()
            else:
                # Face not recognized
                message_placeholder.error("❌ Face Not Recognized")
                time.sleep(2)
                message_placeholder.empty()

            # Reset global embedding after every attempt so next person can be recognized
            import Mark_Attendance_Camera
            Mark_Attendance_Camera.attendance_mean_embedding = None

# =================================================
# VIEW ATTENDANCE
# =================================================
elif mode == "View Attendance":
    ATTENDANCE_FILE = os.path.join("Data", "attendance.csv")
    if not os.path.exists(ATTENDANCE_FILE):
        st.warning("No attendance data found.")
    else:
        df_attendance = pd.read_csv(ATTENDANCE_FILE)
        if df_attendance.empty:
            st.warning("No attendance records yet.")
        else:
            st.dataframe(df_attendance)

# =================================================
# UPDATE EMPLOYEE DATA
# =================================================
elif mode == "Update Employee Data":

    st.subheader("Registered Employees")

    if not os.path.exists(EMP_FILE):
        st.warning("No employee data found.")
        st.stop()

    df = pd.read_csv(EMP_FILE)

    if df.empty:
        st.warning("No employees registered yet.")
        st.stop()

    df.reset_index(inplace=True)
    df.rename(columns={"index": "Row_Number"}, inplace=True)

    # ---------------------------
    # SEARCH
    # ---------------------------
    search = st.text_input("Search by Employee ID or Name")

    if search:
        filtered_df = df[
            df["Employee_ID"].astype(str).str.contains(search, case=False, na=False) |
            df["Employee_Name"].astype(str).str.contains(search, case=False, na=False)
        ]

        if not filtered_df.empty:
            selected_row = st.selectbox(
                "Select search result",
                filtered_df["Row_Number"],
                format_func=lambda x: f"Row {x} - {df.loc[x, 'Employee_Name']}"
            )

            df = pd.concat([
                df[df["Row_Number"] == selected_row],
                df[df["Row_Number"] != selected_row]
            ])
        else:
            st.warning("No matching results found.")

    st.markdown("---")

    edited_df = st.data_editor(
        df.drop(columns=["Row_Number"]),
        use_container_width=True,
        num_rows="fixed",
        hide_index=True
    )

    st.markdown("---")

    # ---------------------------
    # UPDATE BUTTON
    # ---------------------------
    if st.button("Update Changes"):

        try:
            for i in range(len(edited_df)):
                old_row = df.iloc[i]
                new_row = edited_df.iloc[i]

                if not old_row.drop("Row_Number").equals(new_row):
                    update_employee(
                        old_row["Employee_ID"],
                        new_row["Employee_ID"],
                        new_row["Employee_Name"],
                        new_row["Phone_Number"],
                        new_row["Email"],
                        new_row["Address"],
                        new_row["Joining_Date"]
                    )

            st.success("Employee data updated successfully! ✅")
            st.rerun()

        except Exception as e:
            st.error(f"Error: {e}")