# 🚀 Deploy to Streamlit Community Cloud

## Prerequisites
✅ GitHub account  
✅ Streamlit Community Cloud account (free at https://streamlit.io/cloud)  
✅ Firebase project configured  
✅ Qdrant Cloud cluster (or local instance)  
✅ OpenAI API key  

---

## Step 1: Push Code to GitHub

### 1.1 Add and Commit Files

```bash
# Navigate to project
cd d:\Projects\Multimodal_rag\MultimodalRag

# Add all files
git add .

# Commit changes
git commit -m "Add Firebase authentication and chat sessions"

# Push to GitHub
git push origin main
```

**Important:** Make sure `.env` and `.streamlit/secrets.toml` are in `.gitignore` (they already are ✅)

---

## Step 2: Deploy to Streamlit Cloud

### 2.1 Sign Up / Sign In
1. Go to https://share.streamlit.io/
2. Sign in with your GitHub account
3. Authorize Streamlit to access your repositories

### 2.2 Create New App
1. Click **"New app"**
2. Select:
   - **Repository:** `<your-github-username>/MultimodalRag` (or your repo name)
   - **Branch:** `main`
   - **Main file path:** `streamlit_lazy.py`
3. Click **"Advanced settings"** before deploying

### 2.3 Configure Python Version (Optional)
- **Python version:** 3.11 (recommended)

---

## Step 3: Configure Secrets

### 3.1 Add Environment Variables

In the **Advanced settings** → **Secrets** section, paste all your environment variables in TOML format:

```toml
# OpenAI Configuration
OPENAI_API_KEY = "sk-proj-..."
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_VISION_MODEL = "gpt-4o-mini"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_RATE_LIMIT = "60"

# Qdrant Cloud Configuration (REQUIRED for deployment)
QDRANT_URL = "https://your-cluster.cloud.qdrant.io"
QDRANT_API_KEY = "your-qdrant-api-key"
QDRANT_COLLECTION_NAME = "multimodal_rag"

# Firebase Configuration
FIREBASE_PROJECT_ID = "multimodalrag-867bd"
FIREBASE_CLIENT_EMAIL = "firebase-adminsdk-xxxxx@multimodalrag-867bd.iam.gserviceaccount.com"
FIREBASE_WEB_API_KEY = "AIzaSy..."

# Firebase Private Key (IMPORTANT: Keep the quotes and line breaks)
FIREBASE_PRIVATE_KEY = """-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhki...
...your-full-private-key...
-----END PRIVATE KEY-----"""

# Application Settings
ALLOW_GENERAL_KNOWLEDGE = "true"
RETRIEVAL_CONFIDENCE_THRESHOLD = "0.3"
MAX_RETRIES = "5"
CHUNK_SIZE = "512"
CHUNK_OVERLAP = "50"
TOP_K_RETRIEVAL = "10"

# PDF Processing
PDF_MAX_PAGES = "100"
PDF_USE_VISION = "true"
PDF_VISION_THRESHOLD = "100"

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "json"
```

### 3.2 Important Notes for Secrets

⚠️ **Critical:**
- **FIREBASE_PRIVATE_KEY:** Must use triple quotes `"""..."""` with actual line breaks preserved
- **QDRANT_URL:** Must use Qdrant Cloud URL (local `localhost` won't work in cloud deployment)
- **No local file paths:** Streamlit Cloud is ephemeral - data doesn't persist between restarts

---

## Step 4: Deploy! 🎉

1. Click **"Deploy!"**
2. Wait 2-5 minutes for initial deployment
3. Your app will be live at: `https://your-app-name.streamlit.app`

---

## Step 5: Test Your Deployment

### ✅ Checklist
- [ ] App loads without errors
- [ ] Firebase authentication works (can login)
- [ ] Chat sessions are created
- [ ] Can ask questions (with general knowledge)
- [ ] Can ingest documents (PDF upload works)
- [ ] Chat history persists across sessions

### 🐛 Troubleshooting

**"Firebase not configured" error:**
- Check secrets formatting (especially `FIREBASE_PRIVATE_KEY` with triple quotes)
- Verify all Firebase variables are set correctly

**"Qdrant connection failed" error:**
- Ensure you're using **Qdrant Cloud URL**, not `localhost`
- Verify `QDRANT_API_KEY` is correct
- Check if Qdrant cluster is active

**"Missing OPENAI_API_KEY" error:**
- Add `OPENAI_API_KEY` to Streamlit Cloud secrets
- Format: `OPENAI_API_KEY = "sk-proj-..."`

**PDF processing not working:**
- Check deployment logs (click "Manage app" → "Logs")
- Verify `packages.txt` has `poppler-utils`

---

## Step 6: Manage Secrets (After Deployment)

### Update Secrets
1. Go to https://share.streamlit.io/
2. Click on your app → **"Settings"** (⚙️)
3. Click **"Secrets"**
4. Edit variables in TOML format
5. Click **"Save"**
6. App will automatically restart

---

## Cost Estimates (Monthly)

### Free Tier Usage:
- **Streamlit Cloud:** FREE ✅ (1 app, unlimited public viewers)
- **Firebase:** FREE ✅ (up to 50K reads/writes per day)
- **Qdrant Cloud:** FREE ✅ (1GB cluster)
- **OpenAI:** ~$5-20 (depends on usage)
  - Text queries: ~$0.001 per query (gpt-4o-mini)
  - Document ingestion: ~$0.01 per page with images

### Total: ~$5-20/month (mostly OpenAI API costs)

---

## Security Best Practices

✅ **Never commit:**
- `.env` file
- `.streamlit/secrets.toml`
- Firebase service account JSON
- API keys in code

✅ **Always use:**
- Streamlit Cloud secrets for environment variables
- Environment variables via `os.getenv()`
- `.gitignore` for sensitive files

✅ **Firebase Security:**
- Create users in Firebase Console directly
- Enable Firebase Authentication email enumeration protection
- Set Firestore security rules (limit reads/writes to authenticated users)

---

## Firestore Security Rules (Recommended)

In Firebase Console → Firestore Database → Rules:

```javascript
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    // Users can only access their own data
    match /users/{userId}/{document=**} {
      allow read, write: if request.auth != null && request.auth.uid == userId;
    }
  }
}
```

---

## Next Steps After Deployment

1. **Share your app:** Send the URL to users
2. **Create Firebase users:** Go to Firebase Console → Authentication → Users → Add user
3. **Monitor usage:**
   - Streamlit Cloud: View app analytics
   - Firebase: Check authentication and Firestore usage
   - OpenAI: Monitor API usage at platform.openai.com
4. **Upload documents:** Use the Ingest tab to add PDFs/images
5. **Chat!** Sessions will be saved to Firebase Firestore

---

## Updating Your Deployed App

Every time you push to GitHub `main` branch, Streamlit Cloud will **automatically redeploy** your app:

```bash
# Make changes to your code
git add .
git commit -m "Update feature X"
git push origin main

# Streamlit Cloud automatically redeploys (takes ~2 min)
```

---

## Support & Resources

- **Streamlit Docs:** https://docs.streamlit.io/streamlit-community-cloud
- **Firebase Console:** https://console.firebase.google.com/
- **Qdrant Cloud:** https://cloud.qdrant.io/
- **OpenAI Platform:** https://platform.openai.com/

---

**🎉 Happy Deploying! Your multimodal RAG assistant is ready to go live!**
