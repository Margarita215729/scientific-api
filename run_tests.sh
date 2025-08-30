#!/bin/bash

# Test runner script for Scientific API

echo "🧪 Running Scientific API Tests"
echo "================================"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run tests with coverage
echo "Running tests..."
pytest tests/ -v --cov=api --cov=utils --cov=database --cov-report=term-missing --cov-report=html:htmlcov

# Check test results
if [ $? -eq 0 ]; then
    echo "✅ All tests passed!"
    echo "📊 Coverage report generated in htmlcov/"
else
    echo "❌ Some tests failed!"
    exit 1
fi

# Run specific test categories
echo ""
echo "Running specific test categories..."
echo ""

echo "🔬 Unit tests..."
pytest tests/ -m unit -v

echo "🔗 Integration tests..."
pytest tests/ -m integration -v

echo "🐌 Slow tests..."
pytest tests/ -m slow -v

echo ""
echo "🎉 Test run completed!"