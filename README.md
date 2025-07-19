# QuickNotes

A web application for generating notes, summaries, and questions from various content sources including text, PDFs, and YouTube videos.

## Features

- Upload and process PDF documents
- Extract and summarize YouTube video transcripts
- Generate questions from content
- Download summaries and notes

## Local Development

1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the development server:
   ```
   python app.py
   ```
5. Open http://localhost:5000 in your browser

## Deployment to Render

This application is configured for deployment on [Render](https://render.com/).

### Prerequisites

- A Render account (free tier available)
- A GitHub account with access to this repository

### Deployment Steps

1. **Push your code** to a GitHub repository if you haven't already.

2. **Sign in to Render** at [https://dashboard.render.com/](https://dashboard.render.com/)

3. **Create a new Web Service**
   - Click "New" and select "Web Service"
   - Connect your GitHub repository
   - Configure the service:
     - Name: `quicknotes` (or your preferred name)
     - Region: Choose the one closest to you
     - Branch: `main` (or your default branch)
     - Runtime: `Python 3`
     - Build Command: `pip install -r requirements.txt`
     - Start Command: `gunicorn app:app`

4. **Set Environment Variables**
   - Click on the "Environment" tab
   - Add the following environment variables:
     - `PYTHON_VERSION`: `3.9.16`
     - `SECRET_KEY`: Generate a secure random string (you can use `python -c "import os; print(os.urandom(24).hex())"`)

5. **Deploy**
   - Click "Save" and then "Deploy"
   - Wait for the deployment to complete
   - Your app will be live at the URL shown in the Render dashboard

## Configuration

### Environment Variables

- `SECRET_KEY`: Secret key for Flask sessions (required in production)
- `PORT`: Port to run the application on (automatically set by Render)

## File Structure

- `app.py`: Main application file
- `templates/`: HTML templates
- `static/`: Static files (CSS, JS, images)
- `uploads/`: Directory for uploaded files (created automatically)
- `requirements.txt`: Python dependencies
- `Procfile`: Process file for Render
- `runtime.txt`: Python version for Render
- `render.yaml`: Render deployment configuration

## License

This project is open source and available under the [MIT License](LICENSE).