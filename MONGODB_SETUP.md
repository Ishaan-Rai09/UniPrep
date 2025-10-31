# MongoDB Setup Guide

## Overview
The UniPrep app now includes automatic MongoDB integration to save your MCQ history. All you need to do is add your MongoDB credentials to the `.env` file!

## Quick Setup

### 1. Edit the `.env` file
Open the `.env` file in the project root and add your MongoDB URI:

```env
# MongoDB Configuration
MONGODB_URI=mongodb://localhost:27017/
```

### 2. Choose Your MongoDB Option

#### Option A: Local MongoDB (Recommended for Development)
```env
MONGODB_URI=mongodb://localhost:27017/
```
- Install MongoDB locally: https://www.mongodb.com/try/download/community
- Default port is 27017
- No additional configuration needed

#### Option B: MongoDB Atlas (Cloud - Free Tier Available)
```env
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/uniprep_db
```
1. Sign up at https://www.mongodb.com/cloud/atlas
2. Create a free cluster
3. Click "Connect" → "Connect your application"
4. Copy the connection string
5. Replace `<username>` and `<password>` with your credentials

### 3. Restart the Application
```bash
streamlit run app.py
```

## Features

### Automatic Connection
- The app automatically connects to MongoDB when it starts
- No need to manually click "Connect"
- Connection status shows in the sidebar

### User Identification
- Enter your name or ID in the sidebar under "MongoDB Storage"
- All your MCQs will be saved under this ID
- You can change it anytime

### My PDF Questions
- View all your previously generated MCQs
- See metadata: PDF name, difficulty, question count, date
- Click "View Questions" to review saved MCQs
- Refresh button to update the list

### What Gets Saved?
Every time you generate MCQs, the following is saved:
- User ID
- PDF filename
- Complete MCQ text
- Number of questions
- Difficulty level
- Timestamp

## Database Structure

**Database Name:** `uniprep_db`

**Collection:** `mcq_history`

**Document Example:**
```json
{
  "_id": "...",
  "user_id": "John Doe",
  "pdf_name": "Computer_Science_Notes.pdf",
  "mcqs_text": "Question 1: ...",
  "num_questions": 5,
  "difficulty": "Hard",
  "created_at": ISODate("2025-10-30T18:30:00Z"),
  "timestamp": "2025-10-30T18:30:00"
}
```

## Troubleshooting

### Connection Failed
- **Check .env file:** Make sure `MONGODB_URI` is correctly set
- **Local MongoDB:** Ensure MongoDB service is running
- **Atlas:** Verify username, password, and cluster URL
- **Network:** Check if your network allows MongoDB connections

### Not Saving MCQs
- Check the sidebar for connection status
- Look for "✅ Storage Active" message
- Restart the app after updating .env

### Can't See History
- Make sure you're using the same User ID
- Check MongoDB connection status
- Try clicking the "🔄 Refresh" button

## Optional: Without MongoDB
If you don't set up MongoDB:
- The app works perfectly fine
- You can still generate and download MCQs
- History feature will be unavailable
- You'll see "MongoDB not connected" in the sidebar

## Privacy & Security
- Your MongoDB credentials stay in the `.env` file (never committed to git)
- Only you can access your saved MCQs
- User IDs are simple identifiers (not authentication)
- For production use, implement proper authentication

## Support
For MongoDB-specific issues:
- MongoDB Community: https://www.mongodb.com/community
- MongoDB Atlas Docs: https://docs.atlas.mongodb.com/
