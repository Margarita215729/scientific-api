# Scientific API - Astronomical Data Analysis Platform

A comprehensive platform for astronomical data analysis, machine learning, and scientific publication search.

## 🌟 Features

- **Astronomical Catalogs**: Access to SDSS, DESI, DES, and Euclid data
- **ADS Integration**: Search NASA Astrophysics Data System for publications
- **Machine Learning**: Prepare datasets and train models on astronomical data
- **Database Integration**: Support for MongoDB, PostgreSQL, and SQLite
- **Real-time Processing**: Background data processing and model training
- **RESTful API**: Complete API with comprehensive documentation

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd scientific-api
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Initialize database**
   ```bash
   python init_database.py
   ```

### Running the Application

#### Development Mode
```bash
# Start the main backend
python main_azure_with_db.py

# Start the frontend proxy (in another terminal)
python main.py
```

#### Production Mode
```bash
# Using Docker
docker-compose up -d

# Or using the build script
./build_project.sh
```

## 📚 API Documentation

Once the application is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Frontend**: http://localhost:3000

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# Database Configuration
DB_TYPE=sqlite  # Options: sqlite, postgresql, cosmosdb_mongo
DATABASE_URL=sqlite:///./scientific_api.db

# For MongoDB/CosmosDB
AZURE_COSMOS_CONNECTION_STRING=your_connection_string
COSMOS_DATABASE_NAME=scientificdata

# For PostgreSQL
DATABASE_URL=postgresql://user:password@localhost/scientific_api

# ADS API (Optional)
ADSABS_TOKEN=your_ads_token

# Application Settings
HEAVY_PIPELINE_ON_START=false
DEBUG_RELOAD=false
WEB_CONCURRENCY=1
```

### Database Setup

The application supports multiple database backends:

1. **SQLite** (Default): No additional setup required
2. **PostgreSQL**: Install and configure PostgreSQL
3. **MongoDB/CosmosDB**: Set up MongoDB or Azure CosmosDB

## 🧪 Testing

### Run All Tests
```bash
./run_tests.sh
```

### Run Specific Tests
```bash
# Unit tests only
pytest tests/ -m unit

# Integration tests
pytest tests/ -m integration

# With coverage
pytest tests/ --cov=api --cov=utils --cov=database --cov-report=html
```

### Test Categories
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end workflow testing
- **Slow Tests**: Long-running operations

## 📊 Data Processing

### Astronomical Catalogs

The platform supports the following catalogs:

1. **SDSS DR17**: Sloan Digital Sky Survey Data Release 17
2. **DESI DR1**: Dark Energy Spectroscopic Instrument Data Release 1
3. **DES Y6**: Dark Energy Survey Year 6
4. **Euclid Q1**: Euclid Mission Q1 Data

### Data Pipeline

1. **Download**: Fetch data from official sources
2. **Process**: Clean and standardize data
3. **Merge**: Combine multiple catalogs
4. **Store**: Save to database
5. **Analyze**: Generate statistics and insights

### Machine Learning

The ML pipeline includes:

1. **Data Preparation**: Feature engineering and preprocessing
2. **Model Training**: Support for multiple algorithms
3. **Evaluation**: Comprehensive metrics and validation
4. **Prediction**: Real-time inference capabilities

## 🔍 API Endpoints

### Health & Status
- `GET /ping` - Basic health check
- `GET /api/health` - Comprehensive health status
- `GET /api/database/status` - Database status

### Astronomical Data
- `GET /api/astro/catalogs` - List available catalogs
- `GET /api/astro/statistics` - Get comprehensive statistics
- `GET /api/astro/filter` - Filter galaxies by criteria
- `POST /api/astro/download` - Start catalog download

### ADS Search
- `GET /api/ads/search-by-coordinates` - Search by coordinates
- `GET /api/ads/search-by-object` - Search by object name
- `GET /api/ads/search-by-catalog` - Search by catalog
- `GET /api/ads/citations` - Get paper citations

### Machine Learning
- `POST /api/ml/prepare-dataset` - Prepare ML dataset
- `POST /api/ml/train-model` - Train ML model
- `POST /api/ml/predict` - Make predictions
- `GET /api/ml/models` - List trained models
- `GET /api/ml/dataset-statistics` - Get dataset statistics

### Database Operations
- `GET /api/database/objects` - Get astronomical objects
- `POST /api/database/cache` - Cache API responses
- `GET /api/database/cache/{key}` - Retrieve cached data

## 🏗️ Architecture

### Components

1. **Frontend**: HTML/CSS/JavaScript interface
2. **API Gateway**: FastAPI proxy for routing
3. **Backend**: Main application with business logic
4. **Database**: Data persistence layer
5. **ML Pipeline**: Machine learning processing
6. **Data Pipeline**: Astronomical data processing

### Data Flow

```
User Request → API Gateway → Backend → Database/External APIs → Response
```

## 🚀 Deployment

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# Or build manually
docker build -t scientific-api .
docker run -p 8000:8000 scientific-api
```

### Azure Deployment

```bash
# Deploy to Azure Web App
./deploy_azure.sh

# Or using Bicep
./deploy_azure_bicep.sh
```

### Vercel Deployment

```bash
# Deploy frontend to Vercel
vercel --prod
```

## 🔧 Development

### Project Structure

```
scientific-api/
├── api/                    # API modules
│   ├── astro_catalog_api.py
│   ├── ads_api.py
│   ├── ml_analysis_api.py
│   └── heavy_api.py
├── utils/                  # Utility modules
│   ├── data_preprocessor.py
│   ├── ml_processor.py
│   └── astronomy_catalogs_real.py
├── database/              # Database configuration
│   ├── config.py
│   └── schema.sql
├── ui/                    # Frontend files
│   ├── index.html
│   ├── script.js
│   └── style.css
├── tests/                 # Test files
│   ├── test_api.py
│   └── test_ml_processor.py
├── main.py               # Frontend proxy
├── main_azure_with_db.py # Main backend
└── requirements.txt      # Dependencies
```

### Adding New Features

1. **API Endpoints**: Add to appropriate API module
2. **Data Processing**: Extend utility modules
3. **ML Models**: Add to ML processor
4. **Tests**: Create corresponding test files

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Add docstrings
- Write comprehensive tests

## 📈 Monitoring & Logging

### Logging Configuration

The application uses structured logging with different levels:
- **INFO**: General application flow
- **WARNING**: Non-critical issues
- **ERROR**: Errors that need attention
- **CRITICAL**: System-breaking issues

### Health Monitoring

Monitor application health through:
- `/api/health` endpoint
- Database connectivity checks
- External API status

## 🔒 Security

### API Security

- Input validation on all endpoints
- Rate limiting for external APIs
- Secure database connections
- Environment variable protection

### Data Privacy

- No sensitive data logging
- Secure API token handling
- Database access controls

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Check code style
flake8 api/ utils/ database/

# Run type checking
mypy api/ utils/ database/
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue on GitHub
- Check the documentation
- Review the API docs at `/docs`

## 🔄 Changelog

### Version 2.1.0
- Added comprehensive ML pipeline
- Improved data processing
- Enhanced API documentation
- Added extensive test coverage
- Translated to English
- Fixed import issues
- Added real data processing capabilities

### Version 2.0.0
- Database integration
- Background processing
- Enhanced UI
- API improvements

### Version 1.0.0
- Initial release
- Basic API functionality
- Simple UI