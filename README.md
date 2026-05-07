# 📋 Facial Recognition Attendance System

A modern, web-based facial recognition attendance system built with Streamlit. This system uses advanced face detection and deep learning embeddings to automatically register employees and mark their attendance.

## Features

### 🔐 Employee Registration
- Register new employees with personal information (ID, name, phone, email, address)
- Capture multiple face samples using your webcam
- Automatically generates face embeddings for accurate recognition
- Stores employee data and face embeddings in CSV files

### ✅ Attendance Marking
- Real-time facial recognition to mark attendance
- Captures live video feed and matches faces against the employee database
- Automatic timestamp recording
- Support for marking multiple employees' attendance in one session
- Similarity score-based matching for accurate identification

### 📊 View & Manage Data
- View all recorded attendance records with dates and times
- Update existing employee information
- Browse employee database
- Export attendance data as needed

## Technology Stack

### Core Libraries
- **Streamlit**: Web framework for the user interface
- **OpenCV**: Computer vision and video processing
- **MTCNN**: Face detection (Multi-task Cascaded Convolutional Networks)
- **FaceNet (Keras)**: Face embedding generation
- **TensorFlow**: Deep learning backend
- **NumPy & Pandas**: Data processing and manipulation

### Python Version
- Python 3.11.0

## Installation

### Prerequisites
- Python 3.11.0
- Webcam/Camera device
- 500+ MB disk space for dependencies

### Setup Steps

1. **Clone or download the repository**
   ```bash
   cd "Attendance System"
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # On Windows
   # or
   source venv/bin/activate      # On macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Access the web interface**
   - Open your browser to `http://localhost:8501`

## Project Structure

```
├── app.py                          # Main Streamlit application
├── csv_storage.py                  # CSV file management utilities
├── Embedding_Matcher.py            # Face matching logic
├── Register_Camera.py              # Employee registration camera module
├── Mark_Attendance_Camera.py       # Attendance marking camera module
├── requirements.txt                # Python dependencies
├── runtime.txt                     # Python version specification
├── README.md                       # Project documentation
└── Data/
    ├── employees_info.csv         # Employee records
    ├── employees_embeddings.csv   # Face embeddings
    └── attendance.csv             # Attendance records
```

## Usage Guide

### 📝 Register a New Employee

1. Launch the application and select **"Register Employee"**
2. Fill in the employee details:
   - Employee ID (e.g., E101)
   - Employee Name
   - Phone Number
   - Email
   - Address
   - Registration Date
3. Click **"📷 Open Camera"** button
4. Look at the camera and remain still
5. The system captures 10 face samples automatically
6. Once complete, click **"Save Employee"** to store the data

### ✅ Mark Attendance

1. Select **"Mark Attendance"** from the main menu
2. Click **"📷 Open Camera"** to start the camera
3. Look at the camera and hold steady
4. The system captures 5 face samples
5. The captured face is matched against the employee database
6. If a match is found, attendance is automatically marked with:
   - Employee ID
   - Employee Name
   - Current Date & Time
   - Similarity Score
7. Process repeats for marking multiple employees

### 📊 View Attendance Records

1. Select **"View Attendance"** from the main menu
2. Browse all recorded attendance entries
3. Filter or search through records as needed
4. Export data if required (CSV format)

### ✏️ Update Employee Data

1. Select **"Update Employee Data"** from the main menu
2. Enter the Employee ID to search
3. Modify the desired information
4. Save the changes

## How It Works

### Face Detection & Embedding
- **MTCNN** detects faces in video frames
- **FaceNet** generates 128-dimensional face embeddings (numerical representations of unique facial features)
- Multiple embeddings are averaged to create a robust face signature

### Face Matching
- Uses **Cosine Similarity** to compare embeddings
- Calculates similarity score between captured face and all employees in database
- Identifies the employee with the highest similarity score (above threshold)
- Records attendance if match confidence is sufficient

### Data Storage
- All data stored in CSV format for easy access and portability
- Employees info: ID, name, contact, address, joining date
- Embeddings: Employee ID with their 128-dimensional face embedding
- Attendance: Employee ID, name, date, time, and similarity score

## Performance Considerations

- **First run**: Expect longer load time (TensorFlow model initialization)
- **Camera capture**: 10 frames for registration, 5 frames for attendance marking
- **Face matching**: Real-time processing with instant results
- **Database**: CSV-based, suitable for 100-1000+ employees

## Troubleshooting

### Camera Not Working
- Check if your webcam is connected and recognized by the system
- Ensure no other applications are using the camera
- Try refreshing the browser or restarting the application

### Face Not Detected
- Ensure adequate lighting in the environment
- Position your face clearly in front of the camera
- Make sure your face is fully visible without obstructions

### No Match Found During Attendance
- Employee may not be registered yet
- Lighting conditions may be different from registration
- Try getting closer to the camera
- Consider re-registering with better lighting

### Slow Performance
- Close unnecessary applications to free up RAM
- Reduce other resource-intensive processes
- Consider optimizing the employee database size

## Future Enhancements

- Database integration (PostgreSQL/MongoDB)
- Mobile application support
- Advanced analytics and reporting
- Mask detection and handling
- Multi-face simultaneous detection
- Scheduled automated reports
- Integration with HR systems
- Real-time notifications
- Liveness detection for security

## Security Notes

- Store the Data folder in a secure location
- Restrict access to attendance records
- Regularly backup attendance data
- Consider encrypting embeddings for sensitive deployments
- Use HTTPS for production deployments

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 2 GB | 4 GB+ |
| Storage | 500 MB | 1 GB+ |
| Processor | Dual Core | Quad Core+ |
| Camera | Any USB/Built-in | HD or better |

## Dependencies

See `requirements.txt` for all Python package dependencies:
- streamlit 1.56.0
- opencv-python-headless 4.11.0.86
- numpy 1.23.5
- pandas 2.3.3
- mtcnn 1.0.0
- keras-facenet 0.3.2
- tensorflow 2.12.0
- scipy 1.15.3

## License

[Add your license information here]

## Contributing

Contributions are welcome! Please feel free to submit issues and enhancement requests.

## Support

For issues, questions, or suggestions, please open an issue in the repository.

## Disclaimer

This system is designed for general attendance tracking purposes. For deployment in critical applications, additional security measures and testing are recommended.

---

**Built with ❤️ using Streamlit & FaceNet**
