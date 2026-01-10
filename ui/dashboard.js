// Streamlined dashboard functionality - main coordination
// Data collection functionality moved to data-collection.js
// Modal management moved to modal-utils.js
// Notifications moved to notification-utils.js

const API_BASE_URL = '/api';

// Memoization cache for expensive operations
const cache = new Map();

// Cached memoization function
function memoize(fn, keyFn = (...args) => JSON.stringify(args)) {
    return function(...args) {
        const key = keyFn(...args);
        if (cache.has(key)) {
            return cache.get(key);
        }
        const result = fn.apply(this, args);
        cache.set(key, result);
        return result;
    };
}

// Main data collection entry point - delegates to data-collection.js
function showDataCollection() {
    dataCollectionManager.showDataCollection();
}

// Data cleaning function
function runDataCleaning() {
    modalManager.closeModal('dataCleaningModal');
    notificationManager.show('info', 'Processing data...', 'Running cleaning algorithms');
    
    // Simulate processing
    setTimeout(() => {
        notificationManager.show('success', 'Data cleaned!', 'Processed 1,234 records successfully');
    }, 3000);
}

// Data cleaning interface
function showDataCleaning() {
    const content = `
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <!-- Data Issues -->
            <div>
                <h4 class="text-lg font-semibold text-gray-800 mb-3">Detected Issues</h4>
                <div class="space-y-3">
                    <div class="p-3 bg-red-50 border border-red-200 rounded-lg">
                        <div class="flex items-center">
                            <i class="fas fa-exclamation-circle text-red-600 mr-2" aria-hidden="true"></i>
                            <span class="font-medium text-red-800">Missing Values</span>
                        </div>
                        <p class="text-sm text-red-700 mt-1">234 rows with null values in 'magnitude' column</p>
                    </div>
                    
                    <div class="p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
                        <div class="flex items-center">
                            <i class="fas fa-exclamation-triangle text-yellow-600 mr-2" aria-hidden="true"></i>
                            <span class="font-medium text-yellow-800">Duplicate Entries</span>
                        </div>
                        <p class="text-sm text-yellow-700 mt-1">89 potential duplicate records found</p>
                    </div>
                    
                    <div class="p-3 bg-blue-50 border border-blue-200 rounded-lg">
                        <div class="flex items-center">
                            <i class="fas fa-info-circle text-blue-600 mr-2" aria-hidden="true"></i>
                            <span class="font-medium text-blue-800">Format Issues</span>
                        </div>
                        <p class="text-sm text-blue-700 mt-1">Date format inconsistencies in 12 records</p>
                    </div>
                </div>
            </div>
            
            <!-- Cleaning Actions -->
            <div>
                <h4 class="text-lg font-semibold text-gray-800 mb-3">Recommended Actions</h4>
                <div class="space-y-3">
                    <button class="w-full text-left p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-all">
                        <div class="flex justify-between items-center">
                            <div>
                                <p class="font-medium text-gray-900">Handle Missing Values</p>
                                <p class="text-sm text-gray-600">Remove or interpolate missing data</p>
                            </div>
                            <i class="fas fa-arrow-right text-gray-400" aria-hidden="true"></i>
                        </div>
                    </button>
                    
                    <button class="w-full text-left p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-all">
                        <div class="flex justify-between items-center">
                            <div>
                                <p class="font-medium text-gray-900">Remove Duplicates</p>
                                <p class="text-sm text-gray-600">Automatically detect and remove duplicates</p>
                            </div>
                            <i class="fas fa-arrow-right text-gray-400" aria-hidden="true"></i>
                        </div>
                    </button>
                    
                    <button class="w-full text-left p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-all">
                        <div class="flex justify-between items-center">
                            <div>
                                <p class="font-medium text-gray-900">Standardize Formats</p>
                                <p class="text-sm text-gray-600">Convert dates and units to consistent format</p>
                            </div>
                            <i class="fas fa-arrow-right text-gray-400" aria-hidden="true"></i>
                        </div>
                    </button>
                </div>
            </div>
        </div>
    `;

    modalManager.showModal({
        id: 'dataCleaningModal',
        title: 'Data Cleaning & Transformation',
        content: content,
        size: 'xlarge',
        actions: [
            {
                text: 'Cancel',
                class: 'border border-gray-300 text-gray-700 hover:bg-gray-50',
                onclick: "modalManager.closeModal('dataCleaningModal')"
            },
            {
                text: 'Run Cleaning',
                class: 'bg-indigo-600 text-white hover:bg-indigo-700',
                onclick: "runDataCleaning()"
            }
        ]
    });
}

// Show analysis interface
function showAnalysis() {
    const content = `
        <div class="space-y-4">
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div class="p-4 border border-gray-200 rounded-lg hover:border-indigo-300 cursor-pointer">
                    <h4 class="font-semibold text-gray-900 mb-2">Statistical Analysis</h4>
                    <p class="text-sm text-gray-600">Descriptive statistics, correlations, and distributions</p>
                </div>
                <div class="p-4 border border-gray-200 rounded-lg hover:border-indigo-300 cursor-pointer">
                    <h4 class="font-semibold text-gray-900 mb-2">Machine Learning</h4>
                    <p class="text-sm text-gray-600">Train models for classification and regression</p>
                </div>
                <div class="p-4 border border-gray-200 rounded-lg hover:border-indigo-300 cursor-pointer">
                    <h4 class="font-semibold text-gray-900 mb-2">Time Series</h4>
                    <p class="text-sm text-gray-600">Temporal analysis and forecasting</p>
                </div>
                <div class="p-4 border border-gray-200 rounded-lg hover:border-indigo-300 cursor-pointer">
                    <h4 class="font-semibold text-gray-900 mb-2">Custom Analysis</h4>
                    <p class="text-sm text-gray-600">Run custom Python/R scripts</p>
                </div>
            </div>
        </div>
    `;

    modalManager.showModal({
        id: 'analysisModal',
        title: 'Data Analysis Tools',
        content: content,
        size: 'large'
    });
}

// Show visualization interface
function showVisualization() {
    const content = `
        <div class="space-y-4">
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div class="p-4 border border-gray-200 rounded-lg hover:border-indigo-300 cursor-pointer">
                    <i class="fas fa-chart-bar text-2xl text-indigo-600 mb-2" aria-hidden="true"></i>
                    <h4 class="font-semibold text-gray-900 mb-1">Bar Charts</h4>
                    <p class="text-xs text-gray-600">Compare categories</p>
                </div>
                <div class="p-4 border border-gray-200 rounded-lg hover:border-indigo-300 cursor-pointer">
                    <i class="fas fa-chart-line text-2xl text-green-600 mb-2" aria-hidden="true"></i>
                    <h4 class="font-semibold text-gray-900 mb-1">Line Charts</h4>
                    <p class="text-xs text-gray-600">Show trends over time</p>
                </div>
                <div class="p-4 border border-gray-200 rounded-lg hover:border-indigo-300 cursor-pointer">
                    <i class="fas fa-chart-pie text-2xl text-purple-600 mb-2" aria-hidden="true"></i>
                    <h4 class="font-semibold text-gray-900 mb-1">Pie Charts</h4>
                    <p class="text-xs text-gray-600">Show proportions</p>
                </div>
                <div class="p-4 border border-gray-200 rounded-lg hover:border-indigo-300 cursor-pointer">
                    <i class="fas fa-chart-area text-2xl text-red-600 mb-2" aria-hidden="true"></i>
                    <h4 class="font-semibold text-gray-900 mb-1">Scatter Plots</h4>
                    <p class="text-xs text-gray-600">Show correlations</p>
                </div>
                <div class="p-4 border border-gray-200 rounded-lg hover:border-indigo-300 cursor-pointer">
                    <i class="fas fa-globe text-2xl text-blue-600 mb-2" aria-hidden="true"></i>
                    <h4 class="font-semibold text-gray-900 mb-1">Sky Maps</h4>
                    <p class="text-xs text-gray-600">Astronomical coordinates</p>
                </div>
                <div class="p-4 border border-gray-200 rounded-lg hover:border-indigo-300 cursor-pointer">
                    <i class="fas fa-cube text-2xl text-orange-600 mb-2" aria-hidden="true"></i>
                    <h4 class="font-semibold text-gray-900 mb-1">3D Plots</h4>
                    <p class="text-xs text-gray-600">Three-dimensional data</p>
                </div>
            </div>
        </div>
    `;

    modalManager.showModal({
        id: 'visualizationModal',
        title: 'Data Visualization',
        content: content,
        size: 'large'
    });
}

// Efficient file size formatter with memoization
const formatFileSize = memoize((bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
});

// Initialize dashboard with proper cleanup
let dashboardInitialized = false;

function initializeDashboard() {
    if (dashboardInitialized) return;
    
    // Check authentication
    checkAuth();
    
    // Load dashboard data
    loadDashboardData();
    
    dashboardInitialized = true;
}

// Authentication check with caching
const checkAuth = memoize(async () => {
    try {
        // TODO: Implement actual authentication check
        console.log('Authentication check - placeholder');
        return { authenticated: true };
    } catch (error) {
        console.error('Auth check failed:', error);
        return { authenticated: false };
    }
}, () => 'auth_check');

// Load dashboard data with caching
const loadDashboardData = memoize(async () => {
    try {
        // TODO: Load actual dashboard statistics
        console.log('Loading dashboard data - placeholder');
        return { status: 'loaded' };
    } catch (error) {
        console.error('Failed to load dashboard data:', error);
        notificationManager.show('error', 'Failed to load data', 'Please refresh the page');
    }
}, () => 'dashboard_data');

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', initializeDashboard);

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    // Clean up event listeners and cache
    if (dataCollectionManager && dataCollectionManager.cleanup) {
        dataCollectionManager.cleanup();
    }
    cache.clear();
});