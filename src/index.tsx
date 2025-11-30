import { Hono } from 'hono'
import { cors } from 'hono/cors'
import { serveStatic } from 'hono/cloudflare-workers'

const app = new Hono()

// Enable CORS for frontend-backend communication
app.use('/api/*', cors())

// Serve static files from public directory
app.use('/static/*', serveStatic({ root: './public' }))

// Store processing jobs (in production, use a proper database)
let processingJobs = new Map()

// API routes
app.get('/api/hello', (c) => {
  return c.json({ message: 'Hello from Mozart OMR API!' })
})

// Health check
app.get('/api/health', (c) => {
  return c.json({ 
    status: 'healthy',
    service: 'Mozart OMR',
    version: '1.0.0',
    timestamp: new Date().toISOString()
  })
})

// Get dataset information
app.get('/api/dataset', (c) => {
  const datasetInfo = {
    name: "Mozart OMR Dataset",
    version: "1.0",
    description: "Musical symbol dataset for Optical Music Recognition",
    classes: [
      'c1', 'c2', 'd1', 'd2', 'e1', 'e2', 'f1', 'f2', 'g1', 'g2', 'a1', 'a2', 'b1', 'b2',
      'sharp', 'flat', 'natural', 'clef', 'bar', 'dot', 'chord',
      '1', '2', '4', '8', '16', '32'
    ],
    total_classes: 27,
    created: "2024-11-30",
    summary: {
      total_images: 200,
      average_per_class: 7.4,
      training_ready: true
    }
  }
  
  return c.json(datasetInfo)
})

// Upload and process OMR image
app.post('/api/process', async (c) => {
  try {
    const formData = await c.req.formData()
    const imageFile = formData.get('image')
    
    if (!imageFile || !(imageFile instanceof File)) {
      return c.json({ error: 'No image file provided' }, 400)
    }

    // Validate file type
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'application/pdf']
    if (!validTypes.includes(imageFile.type)) {
      return c.json({ error: 'Invalid file type. Please upload JPG, PNG, or PDF files.' }, 400)
    }

    // Validate file size (max 10MB)
    const maxSize = 10 * 1024 * 1024 // 10MB
    if (imageFile.size > maxSize) {
      return c.json({ error: 'File size must be less than 10MB' }, 400)
    }

    // Create processing job
    const jobId = Date.now().toString()
    
    // Simulate OMR processing (in real implementation, call Python backend)
    const result = await simulateOMRProcessing(imageFile, jobId)
    
    return c.json({
      success: true,
      message: 'Image processed successfully',
      jobId: jobId,
      filename: imageFile.name,
      size: imageFile.size,
      type: imageFile.type,
      processingTime: result.processingTime,
      result: result
    })
  } catch (error) {
    console.error('Processing error:', error)
    return c.json({ error: 'Processing failed: ' + error.message }, 500)
  }
})

// Get processing status
app.get('/api/status/:jobId', (c) => {
  const jobId = c.req.param('jobId')
  const job = processingJobs.get(jobId)
  
  if (!job) {
    return c.json({ error: 'Job not found' }, 404)
  }
  
  return c.json({
    jobId: jobId,
    status: job.status,
    progress: job.progress,
    message: job.message,
    result: job.result
  })
})

// Get processing history
app.get('/api/history', (c) => {
  const history = Array.from(processingJobs.values())
    .sort((a, b) => b.timestamp - a.timestamp)
    .slice(0, 10) // Last 10 jobs
  
  return c.json({
    history: history,
    total: processingJobs.size
  })
})

// Download results
app.post('/api/download', async (c) => {
  const { jobId } = await c.req.json()
  const job = processingJobs.get(jobId)
  
  if (!job || !job.result) {
    return c.json({ error: 'Results not found' }, 404)
  }
  
  // Generate downloadable content
  const content = generateResultsContent(job)
  
  return c.json({
    success: true,
    content: content,
    filename: `omr_results_${jobId}.txt`
  })
})

// Simulate OMR processing (replace with actual Python backend call)
async function simulateOMRProcessing(imageFile, jobId) {
  // Simulate processing time
  await new Promise(resolve => setTimeout(resolve, 2000))
  
  // Create processing job
  const job = {
    jobId: jobId,
    status: 'processing',
    progress: 0,
    message: 'Analyzing musical symbols...',
    timestamp: Date.now(),
    result: null
  }
  
  processingJobs.set(jobId, job)
  
  // Simulate processing steps
  const steps = [
    { progress: 20, message: 'Preprocessing image...' },
    { progress: 40, message: 'Detecting staff lines...' },
    { progress: 60, message: 'Extracting symbols...' },
    { progress: 80, message: 'Classifying notes...' },
    { progress: 100, message: 'Processing complete!' }
  ]
  
  // Simulate step-by-step processing
  for (const step of steps) {
    job.progress = step.progress
    job.message = step.message
    await new Promise(resolve => setTimeout(resolve, 400))
  }
  
  // Generate mock results (in real implementation, call Python OMR)
  const mockResults = {
    notes: generateMockNotes(),
    symbols: generateMockSymbols(),
    confidence: Math.random() * 0.3 + 0.7, // 70-100%
    measures: Math.floor(Math.random() * 8) + 4,
    keySignature: ['C major', 'G major', 'D major', 'F major'][Math.floor(Math.random() * 4)],
    timeSignature: ['4/4', '3/4', '2/4', '6/8'][Math.floor(Math.random() * 4)],
    tempo: Math.floor(Math.random() * 60) + 80
  }
  
  job.status = 'completed'
  job.result = mockResults
  job.processingTime = '2.3s'
  
  processingJobs.set(jobId, job)
  
  return mockResults
}

// Generate mock musical notes
function generateMockNotes() {
  const noteNames = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
  const octaves = ['3', '4', '5']
  const notes = []
  
  const numNotes = Math.floor(Math.random() * 20) + 10 // 10-30 notes
  
  for (let i = 0; i < numNotes; i++) {
    const note = noteNames[Math.floor(Math.random() * noteNames.length)]
    const octave = octaves[Math.floor(Math.random() * octaves.length)]
    notes.push(note + octave)
  }
  
  return notes
}

// Generate mock musical symbols
function generateMockSymbols() {
  const symbolTypes = ['sharp', 'flat', 'natural', 'clef', 'bar', 'dot', 'chord']
  const symbols = []
  
  const numSymbols = Math.floor(Math.random() * 15) + 5 // 5-20 symbols
  
  for (let i = 0; i < numSymbols; i++) {
    const symbol = symbolTypes[Math.floor(Math.random() * symbolTypes.length)]
    symbols.push(symbol)
  }
  
  return symbols
}

// Generate downloadable results content
function generateResultsContent(job) {
  const result = job.result
  const timestamp = new Date(job.timestamp).toLocaleString()
  
  return `Mozart OMR Processing Results
============================
Processed File: ${job.filename || 'Unknown'}
Processing Date: ${timestamp}
Processing Time: ${job.processingTime || 'N/A'}

Musical Analysis:
- Key Signature: ${result.keySignature}
- Time Signature: ${result.timeSignature}
- Tempo: ${result.tempo} BPM
- Measures: ${result.measures}

Detected Notes (${result.notes.length}):
${result.notes.join(', ')}

Musical Symbols (${result.symbols.length}):
${result.symbols.join(', ')}

Confidence Score: ${Math.round(result.confidence * 100)}%

Generated by Mozart OMR System
Advanced Optical Music Recognition Technology
`
}

// Default route - serve the main application
app.get('/', (c) => {
  return c.html(`
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Mozart OMR - Optical Music Recognition</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6.4.0/css/all.min.css" rel="stylesheet">
        <link href="/static/styles.css" rel="stylesheet">
    </head>
    <body class="bg-gradient-to-br from-blue-50 to-indigo-100 min-h-screen">
        <div class="container mx-auto px-4 py-8">
            <!-- Header -->
            <header class="text-center mb-12">
                <div class="flex items-center justify-center mb-4">
                    <i class="fas fa-music text-4xl text-indigo-600 mr-3"></i>
                    <h1 class="text-4xl font-bold text-gray-800">
                        Mozart OMR
                    </h1>
                </div>
                <p class="text-xl text-gray-600 mb-6">
                    Transform sheet music into digital format with AI
                </p>
                <div class="flex justify-center space-x-4 text-sm text-gray-500">
                    <span><i class="fas fa-brain mr-1"></i>AI Powered</span>
                    <span><i class="fas fa-bolt mr-1"></i>Lightning Fast</span>
                    <span><i class="fas fa-shield-alt mr-1"></i>Privacy First</span>
                </div>
            </header>

            <!-- Main Content -->
            <main class="max-w-4xl mx-auto">
                <!-- Upload Section -->
                <section class="bg-white rounded-2xl shadow-xl p-8 mb-8">
                    <h2 class="text-2xl font-semibold text-gray-800 mb-6 text-center">
                        Upload Your Sheet Music
                    </h2>
                    
                    <!-- Drag and Drop Area -->
                    <div id="dropZone" class="border-3 border-dashed border-gray-300 rounded-xl p-12 text-center transition-all duration-300 hover:border-indigo-500 hover:bg-indigo-50 cursor-pointer">
                        <div id="uploadContent">
                            <i class="fas fa-cloud-upload-alt text-6xl text-gray-400 mb-4"></i>
                            <h3 class="text-xl font-medium text-gray-700 mb-2">
                                Drop your sheet music here
                            </h3>
                            <p class="text-gray-500 mb-4">
                                or click to browse files
                            </p>
                            <p class="text-sm text-gray-400">
                                Supports JPG, PNG, and PDF files
                            </p>
                        </div>
                        <input type="file" id="fileInput" accept=".jpg,.jpeg,.png,.pdf" class="hidden" />
                    </div>

                    <!-- Preview Area -->
                    <div id="previewArea" class="hidden mt-6">
                        <div class="bg-gray-100 rounded-lg p-4 mb-4">
                            <img id="previewImage" class="max-w-full h-auto mx-auto rounded-lg shadow-md" alt="Preview" />
                        </div>
                        <div class="flex justify-center space-x-4">
                            <button id="processBtn" class="bg-indigo-600 text-white px-6 py-3 rounded-lg hover:bg-indigo-700 transition-colors font-medium">
                                <i class="fas fa-magic mr-2"></i>
                                Process with OMR
                            </button>
                            <button id="clearBtn" class="bg-gray-500 text-white px-6 py-3 rounded-lg hover:bg-gray-600 transition-colors font-medium">
                                <i class="fas fa-times mr-2"></i>
                                Clear
                            </button>
                        </div>
                    </div>
                </section>

                <!-- Processing Status -->
                <section id="processingSection" class="hidden bg-white rounded-2xl shadow-xl p-8 mb-8">
                    <div class="text-center">
                        <div class="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600 mb-4"></div>
                        <h3 class="text-xl font-semibold text-gray-800 mb-2">
                            Processing your sheet music...
                        </h3>
                        <p class="text-gray-600">
                            Our AI is analyzing the musical symbols and notes
                        </p>
                        <div class="mt-4">
                            <div class="w-full bg-gray-200 rounded-full h-2">
                                <div id="progressBar" class="bg-indigo-600 h-2 rounded-full transition-all duration-300" style="width: 0%"></div>
                            </div>
                        </div>
                    </div>
                </section>

                <!-- Results Section -->
                <section id="resultsSection" class="hidden bg-white rounded-2xl shadow-xl p-8">
                    <h2 class="text-2xl font-semibold text-gray-800 mb-6 text-center">
                        Recognition Results
                    </h2>
                    
                    <!-- Results Grid -->
                    <div class="grid md:grid-cols-3 gap-6 mb-8">
                        <div class="bg-green-50 rounded-lg p-6 text-center">
                            <i class="fas fa-music text-3xl text-green-600 mb-3"></i>
                            <h3 class="text-lg font-semibold text-green-800 mb-2">Notes Detected</h3>
                            <p id="notesCount" class="text-2xl font-bold text-green-600">0</p>
                        </div>
                        
                        <div class="bg-blue-50 rounded-lg p-6 text-center">
                            <i class="fas fa-star text-3xl text-blue-600 mb-3"></i>
                            <h3 class="text-lg font-semibold text-blue-800 mb-2">Symbols Found</h3>
                            <p id="symbolsCount" class="text-2xl font-bold text-blue-600">0</p>
                        </div>
                        
                        <div class="bg-purple-50 rounded-lg p-6 text-center">
                            <i class="fas fa-percentage text-3xl text-purple-600 mb-3"></i>
                            <h3 class="text-lg font-semibold text-purple-800 mb-2">Confidence</h3>
                            <p id="confidenceScore" class="text-2xl font-bold text-purple-600">0%</p>
                        </div>
                    </div>

                    <!-- Detailed Results -->
                    <div class="bg-gray-50 rounded-lg p-6">
                        <h3 class="text-lg font-semibold text-gray-800 mb-4">Detected Elements</h3>
                        <div id="detailedResults" class="space-y-2">
                            <!-- Results will be populated here -->
                        </div>
                    </div>

                    <!-- Action Buttons -->
                    <div class="flex justify-center space-x-4 mt-8">
                        <button id="downloadBtn" class="bg-green-600 text-white px-6 py-3 rounded-lg hover:bg-green-700 transition-colors font-medium">
                            <i class="fas fa-download mr-2"></i>
                            Download Results
                        </button>
                        <button id="newProcessBtn" class="bg-indigo-600 text-white px-6 py-3 rounded-lg hover:bg-indigo-700 transition-colors font-medium">
                            <i class="fas fa-plus mr-2"></i>
                            Process Another
                        </button>
                    </div>
                </section>
            </main>

            <!-- Footer -->
            <footer class="text-center mt-12 text-gray-500">
                <p>&copy; 2024 Mozart OMR. Powered by advanced machine learning algorithms.</p>
            </footer>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/axios@1.6.0/dist/axios.min.js"></script>
        <script src="/static/app.js"></script>
    </body>
    </html>
  `)
})

export default app