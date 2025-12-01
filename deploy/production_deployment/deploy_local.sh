#!/bin/bash
echo "🚀 Deploying Government Exam AI API locally..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Build and start services
echo "Building Docker images..."
docker-compose build

echo "Starting services..."
docker-compose up -d

# Wait for services to be ready
echo "Waiting for services to be ready..."
sleep 30

# Test API
echo "Testing API endpoints..."
curl -f http://localhost:8000/health || {
    echo "❌ API health check failed"
    exit 1
}

echo "✅ Deployment successful!"
echo "🌐 API is available at: http://localhost:8000"
echo "📚 API documentation: http://localhost:8000/docs"
echo ""
echo "To stop services: docker-compose down"
