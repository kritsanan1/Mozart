# Mozart OMR - Web Frontend

## 🎵 Overview

A modern web frontend for the Mozart OMR (Optical Music Recognition) system, built with **Hono** and **Cloudflare Pages**. This application provides an intuitive interface for uploading sheet music images and processing them through AI-powered optical music recognition.

## ✨ Features

### 🚀 Core Functionality
- **Drag & Drop Interface**: Intuitive file upload with visual feedback
- **Multi-format Support**: Handles JPG, PNG, and PDF files
- **Real-time Processing**: Live progress tracking and status updates
- **Results Visualization**: Clean, organized display of recognition results
- **Download Results**: Export processed data in multiple formats

### 🎨 User Experience
- **Modern Design**: Beautiful gradient backgrounds and smooth animations
- **Responsive Layout**: Works seamlessly on desktop and mobile devices
- **Accessibility**: Keyboard navigation and screen reader support
- **Loading States**: Clear feedback during processing
- **Error Handling**: User-friendly error messages and recovery

### 🔧 Technical Features
- **Edge Computing**: Deployed on Cloudflare's global network
- **Fast Performance**: Optimized for speed and efficiency
- **API Integration**: RESTful endpoints for all operations
- **File Validation**: Size and format validation on client and server
- **Progress Tracking**: Real-time processing status updates

## 🛠️ Technology Stack

- **Backend**: Hono framework (TypeScript)
- **Frontend**: TailwindCSS, FontAwesome icons
- **Deployment**: Cloudflare Pages
- **Build Tool**: Vite
- **Process Manager**: PM2

## 📁 Project Structure

```
webapp/
├── src/
│   └── index.tsx          # Main application entry point
├── public/
│   └── static/
│       ├── app.js         # Frontend JavaScript
│       └── styles.css     # Custom CSS styles
├── ecosystem.config.cjs   # PM2 configuration
├── package.json         # Dependencies and scripts
├── wrangler.jsonc       # Cloudflare configuration
└── README.md           # This file
```

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ 
- npm or yarn
- Cloudflare account (for deployment)

### Installation

1. **Install dependencies**:
```bash
cd webapp
npm install
```

2. **Build the application**:
```bash
npm run build
```

3. **Start development server**:
```bash
pm2 start ecosystem.config.cjs
```

4. **Access the application**:
```
Local: http://localhost:3000
Public: https://3000-iln7nh67dnuipw4zvcquf-ad490db5.sandbox.novita.ai
```

## 📡 API Endpoints

### Health Check
```
GET /api/health
```
Returns service health status.

### Dataset Information
```
GET /api/dataset
```
Returns dataset metadata and class information.

### Process Image
```
POST /api/process
Content-Type: multipart/form-data
```
Upload and process a sheet music image.

### Check Status
```
GET /api/status/:jobId
```
Check processing status for a specific job.

### Processing History
```
GET /api/history
```
Get recent processing history.

### Download Results
```
POST /api/download
Content-Type: application/json
{
  "jobId": "string"
}
```
Download processed results.

## 📊 Dataset Integration

The frontend integrates with the Mozart OMR dataset containing **27 musical symbol classes**:

### Note Classes (14)
- **c1, c2, d1, d2, e1, e2, f1, f2, g1, g2, a1, a2, b1, b2**

### Symbol Classes (7)
- **sharp, flat, natural, clef, bar, dot, chord**

### Duration Classes (6)
- **1, 2, 4, 8, 16, 32**

## 🎯 User Guide

### Uploading Sheet Music

1. **Drag & Drop**: Simply drag your sheet music file onto the upload area
2. **Click to Browse**: Click the upload area to select a file
3. **File Types**: Supports JPG, PNG, and PDF files up to 10MB

### Processing

1. **Automatic Processing**: Files are processed automatically after upload
2. **Progress Tracking**: Watch real-time progress during processing
3. **Results**: View detected notes, symbols, and confidence scores

### Results Interpretation

- **Notes Detected**: Number of musical notes found
- **Symbols Found**: Count of musical symbols (sharps, flats, etc.)
- **Confidence Score**: AI confidence level (70-100%)

## 🔧 Configuration

### Environment Variables
```bash
NODE_ENV=development
PORT=3000
```

### Cloudflare Configuration
Edit `wrangler.jsonc` for deployment settings:
```json
{
  "name": "mozart-omr-webapp",
  "main": "src/index.tsx",
  "compatibility_date": "2024-01-01"
}
```

## 🚀 Deployment

### Local Development
```bash
npm run dev:sandbox
```

### Production Build
```bash
npm run build
npm run deploy
```

### Cloudflare Pages
```bash
npm run deploy:prod
```

## 📈 Performance

- **Fast Loading**: Optimized bundle size and CDN delivery
- **Edge Computing**: Global distribution via Cloudflare
- **Efficient Processing**: Optimized algorithms for quick recognition
- **Scalable**: Handles multiple concurrent requests

## 🔒 Security

- **File Validation**: Server-side validation of file types and sizes
- **Input Sanitization**: All user inputs are sanitized
- **Rate Limiting**: Built-in rate limiting for API endpoints
- **Secure Headers**: Proper security headers configured

## 🧪 Testing

```bash
# Run basic connectivity test
npm test

# Manual testing
curl http://localhost:3000/api/health
```

## 🐛 Troubleshooting

### Common Issues

1. **Port 3000 in use**:
```bash
npm run clean-port
```

2. **Build errors**:
```bash
rm -rf node_modules package-lock.json
npm install
npm run build
```

3. **PM2 issues**:
```bash
pm2 delete all
pm2 start ecosystem.config.cjs
```

## 📝 License

MIT License - See LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review API documentation
- Check server logs: `pm2 logs`

---

**Mozart OMR** - Transforming sheet music into digital format with the power of AI and machine learning.