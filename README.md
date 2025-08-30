# Scientific Data Management Platform

A comprehensive web platform for collecting, cleaning, analyzing, and visualizing scientific data, with a focus on astronomical datasets.

## 🚀 Features

### ✅ Completed Features

1. **Database Integration**
   - MongoDB Atlas connection configured and tested
   - Support for multiple database backends (MongoDB, PostgreSQL, SQLite)
   - Automatic database initialization and schema management
   - SSL certificate handling for secure connections

2. **Intuitive Web Interface**
   - Modern dashboard with step-by-step workflow
   - Interactive tooltips and help text for all features
   - Responsive design with Tailwind CSS
   - Real-time activity tracking
   - Modal dialogs for complex operations

3. **Comprehensive Data Collection Module**
   - **API Integrations**:
     - SDSS (Sloan Digital Sky Survey) - Galaxy and stellar data
     - NASA Exoplanet Archive - Confirmed exoplanet data
     - ADS (Astrophysics Data System) - Research papers
     - **arXiv** - Latest research preprints
     - **SerpAPI** - Web search and Google Scholar
   - **File Upload**: CSV, JSON, FITS, XML, Parquet with drag-and-drop
   - **Web Scraping**: Configurable scraping capabilities
   - **Database Connections**: Import from external databases
   - Pre-built templates for common datasets
   - Automatic format detection and validation

4. **Advanced Data Cleaning Module**
   - Intelligent issue detection:
     - Missing values analysis with multiple strategies
     - Duplicate detection and removal
     - Outlier identification using IQR method
     - Data type consistency checking
     - Value range validation for scientific data
   - **Cleaning Operations**:
     - Handle missing values (drop, fill with mean/median/mode, interpolate)
     - Remove duplicates with configurable rules
     - Normalize values for ML readiness
     - Remove or cap outliers
     - Standardize formats and data types
   - Preview functionality before applying changes
   - Health score calculation (0-100)
   - Backup creation before cleaning

5. **Robust API Architecture**
   - RESTful endpoints for all operations
   - Background task processing for long operations
   - Comprehensive error handling and logging
   - Integration testing endpoints
   - Real-time status monitoring
   - Template management system

### 🔄 Next Steps

- Machine learning analysis modules (clustering, classification, regression)
- Advanced visualization capabilities (interactive charts, 3D plots)
- User authentication system (OAuth, JWT tokens)
- Production optimization (caching, rate limiting, monitoring)

## 🛠️ Technology Stack

- **Backend**: FastAPI (Python 3.13)
- **Database**: MongoDB Atlas (with Cosmos DB support)
- **Frontend**: HTML5, Tailwind CSS, JavaScript
- **Data Processing**: Pandas, NumPy, Astropy
- **Async Operations**: Motor, httpx, aiohttp

## 📋 Prerequisites

- Python 3.8+
- MongoDB Atlas account or local MongoDB instance
- API keys for external services (optional):
  - ADS (Astrophysics Data System)
  - NASA APIs
  - Google APIs
  - SERPAPI

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/Margarita215729/scientific-api.git
cd scientific-api
```

2. Create virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
# Copy the example config
cp config.env.example config.env

# Edit config.env with your settings:
# - MongoDB connection string
# - API keys
# - Other configuration
```

## 🚀 Running the Application

### Development Mode

```bash
# Using the start script
python start_server.py

# Or directly with uvicorn
uvicorn api.index:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode

```bash
# Using Docker
docker-compose up -d

# Or with gunicorn
gunicorn api.index:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## 📚 API Documentation

Once running, access the interactive API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🎯 Usage Examples

### Import Data from Templates

```javascript
// Import arXiv papers
fetch('/api/data/import-template', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({template: 'arxiv_papers'})
})

// Import from Google Scholar via SerpAPI
fetch('/api/data/import-template', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({template: 'google_scholar'})
})

// Import SDSS galaxy data
fetch('/api/data/import-template', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({template: 'sdss_galaxies'})
})
```

### Upload and Clean Data

```javascript
// Upload CSV/JSON files
const formData = new FormData();
formData.append('files', fileInput.files[0]);

const uploadResponse = await fetch('/api/data/upload', {
    method: 'POST',
    body: formData
});

// Analyze data quality issues
const issuesResponse = await fetch(`/api/cleaning/analyze-issues?dataset_id=${datasetId}`);
const issues = await issuesResponse.json();

// Apply cleaning operations
const cleaningResponse = await fetch('/api/cleaning/apply-cleaning', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        dataset_id: datasetId,
        operations: [
            {operation: 'remove_duplicates'},
            {operation: 'handle_missing', strategy: 'fill_median'},
            {operation: 'normalize_values'}
        ]
    })
});
```

### Test All Integrations

```javascript
// Check status of all data sources
fetch('/api/test/integrations')
    .then(response => response.json())
    .then(data => {
        console.log(`Working sources: ${data.working_sources}/${data.total_sources}`);
        console.log('Recommendations:', data.recommendations);
    });
```

## 🏗️ Project Structure

```
scientific-api/
├── api/                    # API endpoints and routers
│   ├── index.py           # Main FastAPI app
│   ├── heavy_api.py       # Scientific data endpoints
│   ├── data_management.py # Data import/export
│   └── config.py          # Configuration management
├── database/              # Database configuration
│   ├── config.py         # Database connection manager
│   └── schema.sql        # SQL schema definitions
├── utils/                 # Utility modules
│   ├── data_preprocessor.py  # Data preprocessing
│   └── data_processing.py    # Data transformations
├── ui/                    # Frontend files
│   ├── dashboard.html     # Main dashboard
│   └── dashboard.js       # Frontend logic
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🔒 Security Considerations

- Environment variables for sensitive data
- CORS configuration for API access
- Input validation on all endpoints
- Rate limiting for external API calls

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- NASA Open APIs for astronomical data
- SDSS for galaxy survey data
- ADS for research paper access
- MongoDB Atlas for database hosting

## 📞 Support

**Версия**: 2.0.0

📊 **[Подробный статус проекта](PROJECT_STATUS.md)** | 🗓️ **[Дорожная карта развития](ROADMAP.md)** 
=======
For issues and questions:
- Create an issue on GitHub
- Contact: margarita215729@gmail.com