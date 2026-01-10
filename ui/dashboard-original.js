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
                                    <div>
                                        <p class="font-medium">SDSS Galaxy Catalog</p>
                                        <p class="text-xs text-gray-600">Import latest galaxy data from SDSS</p>
                                    </div>
                                    <i class="fas fa-arrow-right text-gray-400"></i>
                                </div>
                            </button>
                            
                            <button onclick="importTemplate('nasa_exoplanets')" class="w-full text-left p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-all">
                                <div class="flex justify-between items-center">
                                    <div>
                                        <p class="font-medium">NASA Exoplanet Archive</p>
                                        <p class="text-xs text-gray-600">Get confirmed exoplanet data</p>
                                    </div>
                                    <i class="fas fa-arrow-right text-gray-400"></i>
                                </div>
                            </button>
                            
                            <button onclick="importTemplate('ads_papers')" class="w-full text-left p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-all">
                                <div class="flex justify-between items-center">
                                    <div>
                                        <p class="font-medium">ADS Research Papers</p>
                                        <p class="text-xs text-gray-600">Search and import paper metadata</p>
                                    </div>
                                    <i class="fas fa-arrow-right text-gray-400"></i>
                                </div>
                            </button>
                            
                            <button onclick="importTemplate('arxiv_papers')" class="w-full text-left p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-all">
                                <div class="flex justify-between items-center">
                                    <div>
                                        <p class="font-medium">arXiv Research Papers</p>
                                        <p class="text-xs text-gray-600">Latest preprints from arXiv</p>
                                    </div>
                                    <i class="fas fa-arrow-right text-gray-400"></i>
                                </div>
                            </button>
                            
                            <button onclick="importTemplate('google_scholar')" class="w-full text-left p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-all">
                                <div class="flex justify-between items-center">
                                    <div>
                                        <p class="font-medium">Google Scholar Papers</p>
                                        <p class="text-xs text-gray-600">Academic papers via SerpAPI</p>
                                    </div>
                                    <i class="fas fa-arrow-right text-gray-400"></i>
                                </div>
                            </button>
                            
                            <button onclick="importTemplate('serpapi_search')" class="w-full text-left p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-all">
                                <div class="flex justify-between items-center">
                                    <div>
                                        <p class="font-medium">Web Search Data</p>
                                        <p class="text-xs text-gray-600">Scientific data from web sources</p>
                                    </div>
                                    <i class="fas fa-arrow-right text-gray-400"></i>
                                </div>
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
    document.getElementById('modalContainer').innerHTML = modal;
}

// Show API import form
function showAPIImport() {
    const modal = `
        <div class="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full z-50" id="apiImportModal">
            <div class="relative top-20 mx-auto p-5 border w-11/12 md:w-3/4 lg:w-1/2 shadow-lg rounded-md bg-white">
                <div class="flex justify-between items-center mb-4">
                    <h3 class="text-xl font-bold text-gray-900">API Data Import</h3>
                    <button onclick="closeModal('apiImportModal')" class="text-gray-400 hover:text-gray-600">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                
                <form onsubmit="handleAPIImport(event)" class="space-y-4">
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">Select API Source</label>
                        <select id="apiSource" class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-indigo-500 focus:border-indigo-500">
                            <option value="">Choose a data source...</option>
                            <option value="sdss">SDSS - Sloan Digital Sky Survey</option>
                            <option value="nasa">NASA Open APIs</option>
                            <option value="ads">ADS - Astrophysics Data System</option>
                            <option value="arxiv">arXiv - Research Preprints</option>
                            <option value="serpapi">SerpAPI - Web Search</option>
                            <option value="google_scholar">Google Scholar (via SerpAPI)</option>
                            <option value="euclid">ESA Euclid Archive</option>
                            <option value="custom">Custom API Endpoint</option>
                        </select>
                    </div>
                    
                    <div id="apiParameters" class="space-y-4">
                        <!-- Dynamic parameters will be loaded here based on selection -->
                    </div>
                    
                    <div class="flex justify-end space-x-3">
                        <button type="button" onclick="closeModal('apiImportModal')" class="px-4 py-2 border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50">
                            Cancel
                        </button>
                        <button type="submit" class="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700">
                            Import Data
                        </button>
                    </div>
                </form>
            </div>
        </div>
    `;
    document.getElementById('modalContainer').innerHTML = modal;
    
    // Add event listener for API source change
    document.getElementById('apiSource').addEventListener('change', loadAPIParameters);
}

// Load API-specific parameters
function loadAPIParameters(event) {
    const source = event.target.value;
    const parametersDiv = document.getElementById('apiParameters');
    
    let parametersHTML = '';
    
    switch(source) {
        case 'sdss':
            parametersHTML = `
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-2">Object Type</label>
                    <select class="w-full px-3 py-2 border border-gray-300 rounded-md">
                        <option value="galaxy">Galaxies</option>
                        <option value="star">Stars</option>
                        <option value="quasar">Quasars</option>
                    </select>
                </div>
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-2">Maximum Records</label>
                    <input type="number" value="1000" min="1" max="100000" class="w-full px-3 py-2 border border-gray-300 rounded-md">
                </div>
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-2">RA Range (degrees)</label>
                    <div class="grid grid-cols-2 gap-2">
                        <input type="number" placeholder="Min RA" step="0.001" class="px-3 py-2 border border-gray-300 rounded-md">
                        <input type="number" placeholder="Max RA" step="0.001" class="px-3 py-2 border border-gray-300 rounded-md">
                    </div>
                </div>
            `;
            break;
            
        case 'ads':
            parametersHTML = `
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-2">Search Query</label>
                    <input type="text" placeholder="e.g., author:Einstein year:2020-2023" class="w-full px-3 py-2 border border-gray-300 rounded-md">
                </div>
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-2">Fields to Include</label>
                    <div class="space-y-2">
                        <label class="flex items-center">
                            <input type="checkbox" checked class="mr-2">
                            <span class="text-sm">Title & Abstract</span>
                        </label>
                        <label class="flex items-center">
                            <input type="checkbox" checked class="mr-2">
                            <span class="text-sm">Authors & Affiliations</span>
                        </label>
                        <label class="flex items-center">
                            <input type="checkbox" class="mr-2">
                            <span class="text-sm">Citations</span>
                        </label>
                    </div>
                </div>
            `;
            break;
            
        case 'arxiv':
            parametersHTML = `
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-2">Search Category</label>
                    <select class="w-full px-3 py-2 border border-gray-300 rounded-md">
                        <option value="cat:astro-ph">Astrophysics</option>
                        <option value="cat:physics">Physics</option>
                        <option value="cat:math">Mathematics</option>
                        <option value="cat:cs">Computer Science</option>
                        <option value="cat:q-bio">Quantitative Biology</option>
                    </select>
                </div>
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-2">Maximum Results</label>
                    <input type="number" value="100" min="1" max="2000" class="w-full px-3 py-2 border border-gray-300 rounded-md">
                </div>
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-2">Sort By</label>
                    <select class="w-full px-3 py-2 border border-gray-300 rounded-md">
                        <option value="submittedDate">Submission Date</option>
                        <option value="lastUpdatedDate">Last Updated</option>
                        <option value="relevance">Relevance</option>
                    </select>
                </div>
            `;
            break;
            
        case 'serpapi':
        case 'google_scholar':
            parametersHTML = `
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-2">Search Query</label>
                    <input type="text" placeholder="e.g., machine learning astronomy" class="w-full px-3 py-2 border border-gray-300 rounded-md">
                </div>
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-2">Number of Results</label>
                    <input type="number" value="50" min="1" max="100" class="w-full px-3 py-2 border border-gray-300 rounded-md">
                </div>
                <div class="grid grid-cols-2 gap-2">
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">Year From</label>
                        <input type="number" value="2020" min="1900" max="2024" class="w-full px-3 py-2 border border-gray-300 rounded-md">
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">Year To</label>
                        <input type="number" value="2024" min="1900" max="2024" class="w-full px-3 py-2 border border-gray-300 rounded-md">
                    </div>
                </div>
                ${source === 'serpapi' ? `
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-2">Search Engine</label>
                    <select class="w-full px-3 py-2 border border-gray-300 rounded-md">
                        <option value="google_scholar">Google Scholar</option>
                        <option value="google">Google Search</option>
                        <option value="bing">Bing</option>
                        <option value="duckduckgo">DuckDuckGo</option>
                    </select>
                </div>
                ` : ''}
            `;
            break;
            
        case 'custom':
            parametersHTML = `
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-2">API Endpoint URL</label>
                    <input type="url" placeholder="https://api.example.com/data" class="w-full px-3 py-2 border border-gray-300 rounded-md">
                </div>
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-2">Authentication</label>
                    <select class="w-full px-3 py-2 border border-gray-300 rounded-md">
                        <option value="none">No Authentication</option>
                        <option value="apikey">API Key</option>
                        <option value="bearer">Bearer Token</option>
                    </select>
                </div>
            `;
            break;
    }
    
    if (source) {
        parametersHTML += `
            <div class="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <div class="flex">
                    <i class="fas fa-info-circle text-blue-600 mr-2"></i>
                    <div>
                        <p class="text-sm text-blue-800 font-semibold">Tips for ${source.toUpperCase()}</p>
                        <p class="text-xs text-blue-700 mt-1">
                            ${getAPITips(source)}
                        </p>
                    </div>
                </div>
            </div>
        `;
    }
    
    parametersDiv.innerHTML = parametersHTML;
}

// Get API-specific tips
function getAPITips(source) {
    const tips = {
        'sdss': 'SDSS contains data for millions of celestial objects. Start with smaller queries to test your filters.',
        'nasa': 'NASA APIs have rate limits. Consider using pagination for large datasets.',
        'ads': 'Use advanced search syntax for precise results. Example: "author:^Smith year:2020"',
        'arxiv': 'arXiv provides free access to research preprints. Use category filters to narrow down results. Example: "cat:astro-ph AND submittedDate:[202001* TO *]"',
        'serpapi': 'SerpAPI provides access to Google Scholar and other search engines. Requires API key. Great for finding research papers and citations.',
        'google_scholar': 'Google Scholar via SerpAPI gives access to academic papers with citation counts. Use specific keywords for better results.',
        'euclid': 'Euclid data may require special permissions. Ensure you have proper access rights.',
        'custom': 'Make sure to handle CORS issues and check API documentation for required headers.'
    };
    return tips[source] || 'Check the API documentation for specific requirements.';
}

// Show file upload interface
function showFileUpload() {
    const modal = `
        <div class="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full z-50" id="fileUploadModal">
            <div class="relative top-20 mx-auto p-5 border w-11/12 md:w-3/4 lg:w-1/2 shadow-lg rounded-md bg-white">
                <div class="flex justify-between items-center mb-4">
                    <h3 class="text-xl font-bold text-gray-900">Upload Data Files</h3>
                    <button onclick="closeModal('fileUploadModal')" class="text-gray-400 hover:text-gray-600">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                
                <div class="space-y-4">
                    <!-- Drag and Drop Area -->
                    <div id="dropZone" class="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-indigo-500 transition-all">
                        <i class="fas fa-cloud-upload-alt text-4xl text-gray-400 mb-4"></i>
                        <p class="text-gray-600 mb-2">Drag and drop files here or click to browse</p>
                        <p class="text-xs text-gray-500">Supported formats: CSV, JSON, FITS, TXT, XML, Parquet</p>
                        <input type="file" id="fileInput" multiple accept=".csv,.json,.fits,.txt,.xml,.parquet" class="hidden">
                        <button onclick="document.getElementById('fileInput').click()" class="mt-4 px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700">
                            Select Files
                        </button>
                    </div>
                    
                    <!-- File Preview -->
                    <div id="filePreview" class="hidden">
                        <h4 class="text-sm font-semibold text-gray-700 mb-2">Selected Files</h4>
                        <div id="fileList" class="space-y-2"></div>
                    </div>
                    
                    <!-- Upload Options -->
                    <div class="space-y-3">
                        <label class="flex items-center">
                            <input type="checkbox" checked class="mr-2">
                            <span class="text-sm">Auto-detect data format and structure</span>
                        </label>
                        <label class="flex items-center">
                            <input type="checkbox" checked class="mr-2">
                            <span class="text-sm">Validate data during import</span>
                        </label>
                        <label class="flex items-center">
                            <input type="checkbox" class="mr-2">
                            <span class="text-sm">Create backup of original files</span>
                        </label>
                    </div>
                    
                    <div class="flex justify-end space-x-3">
                        <button onclick="closeModal('fileUploadModal')" class="px-4 py-2 border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50">
                            Cancel
                        </button>
                        <button onclick="uploadFiles()" class="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700">
                            Upload & Process
                        </button>
                    </div>
                </div>
            </div>
        </div>
    `;
    document.getElementById('modalContainer').innerHTML = modal;
    setupDragAndDrop();
}

// Setup drag and drop functionality
function setupDragAndDrop() {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    
    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });
    
    // Highlight drop zone when item is dragged over it
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });
    
    // Handle dropped files
    dropZone.addEventListener('drop', handleDrop, false);
    fileInput.addEventListener('change', handleFileSelect, false);
}

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function highlight(e) {
    document.getElementById('dropZone').classList.add('border-indigo-500', 'bg-indigo-50');
}

function unhighlight(e) {
    document.getElementById('dropZone').classList.remove('border-indigo-500', 'bg-indigo-50');
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFiles(files);
}

function handleFileSelect(e) {
    const files = e.target.files;
    handleFiles(files);
}

function handleFiles(files) {
    document.getElementById('filePreview').classList.remove('hidden');
    const fileList = document.getElementById('fileList');
    fileList.innerHTML = '';
    
    ([...files]).forEach(file => {
        const fileItem = document.createElement('div');
        fileItem.className = 'flex justify-between items-center p-3 bg-gray-50 rounded-lg';
        fileItem.innerHTML = `
            <div class="flex items-center">
                <i class="fas fa-file text-gray-400 mr-3"></i>
                <div>
                    <p class="text-sm font-medium">${file.name}</p>
                    <p class="text-xs text-gray-500">${formatFileSize(file.size)}</p>
                </div>
            </div>
            <button onclick="removeFile(this)" class="text-red-500 hover:text-red-700">
                <i class="fas fa-times"></i>
            </button>
        `;
        fileList.appendChild(fileItem);
    });
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Data cleaning interface
function showDataCleaning() {
    const modal = `
        <div class="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full z-50" id="dataCleaningModal">
            <div class="relative top-20 mx-auto p-5 border w-11/12 md:w-3/4 lg:w-2/3 shadow-lg rounded-md bg-white">
                <div class="flex justify-between items-center mb-4">
                    <h3 class="text-xl font-bold text-gray-900">Data Cleaning & Transformation</h3>
                    <button onclick="closeModal('dataCleaningModal')" class="text-gray-400 hover:text-gray-600">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <!-- Data Issues -->
                    <div>
                        <h4 class="text-lg font-semibold text-gray-800 mb-3">Detected Issues</h4>
                        <div class="space-y-3">
                            <div class="p-3 bg-red-50 border border-red-200 rounded-lg">
                                <div class="flex justify-between items-center">
                                    <div>
                                        <p class="font-medium text-red-800">Missing Values</p>
                                        <p class="text-sm text-red-600">2,341 cells (3.2%)</p>
                                    </div>
                                    <button class="px-3 py-1 bg-red-600 text-white text-sm rounded hover:bg-red-700">
                                        Fix
                                    </button>
                                </div>
                            </div>
                            
                            <div class="p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
                                <div class="flex justify-between items-center">
                                    <div>
                                        <p class="font-medium text-yellow-800">Duplicates</p>
                                        <p class="text-sm text-yellow-600">156 rows</p>
                                    </div>
                                    <button class="px-3 py-1 bg-yellow-600 text-white text-sm rounded hover:bg-yellow-700">
                                        Remove
                                    </button>
                                </div>
                            </div>
                            
                            <div class="p-3 bg-orange-50 border border-orange-200 rounded-lg">
                                <div class="flex justify-between items-center">
                                    <div>
                                        <p class="font-medium text-orange-800">Format Issues</p>
                                        <p class="text-sm text-orange-600">Date formats inconsistent</p>
                                    </div>
                                    <button class="px-3 py-1 bg-orange-600 text-white text-sm rounded hover:bg-orange-700">
                                        Standardize
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Cleaning Options -->
                    <div>
                        <h4 class="text-lg font-semibold text-gray-800 mb-3">Cleaning Actions</h4>
                        <div class="space-y-3">
                            <div class="border rounded-lg p-3">
                                <label class="flex items-start">
                                    <input type="checkbox" checked class="mt-1 mr-3">
                                    <div>
                                        <p class="font-medium">Handle Missing Values</p>
                                        <p class="text-sm text-gray-600">Fill with mean/median or remove rows</p>
                                    </div>
                                </label>
                            </div>
                            
                            <div class="border rounded-lg p-3">
                                <label class="flex items-start">
                                    <input type="checkbox" checked class="mt-1 mr-3">
                                    <div>
                                        <p class="font-medium">Remove Duplicates</p>
                                        <p class="text-sm text-gray-600">Keep first occurrence only</p>
                                    </div>
                                </label>
                            </div>
                            
                            <div class="border rounded-lg p-3">
                                <label class="flex items-start">
                                    <input type="checkbox" checked class="mt-1 mr-3">
                                    <div>
                                        <p class="font-medium">Normalize Values</p>
                                        <p class="text-sm text-gray-600">Scale numerical features to 0-1 range</p>
                                    </div>
                                </label>
                            </div>
                            
                            <div class="border rounded-lg p-3">
                                <label class="flex items-start">
                                    <input type="checkbox" class="mt-1 mr-3">
                                    <div>
                                        <p class="font-medium">Remove Outliers</p>
                                        <p class="text-sm text-gray-600">Use IQR method (1.5x)</p>
                                    </div>
                                </label>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="mt-6 flex justify-between items-center">
                    <div class="flex items-center text-sm text-gray-600">
                        <i class="fas fa-info-circle mr-2"></i>
                        <span>Preview changes before applying</span>
                    </div>
                    <div class="flex space-x-3">
                        <button onclick="closeModal('dataCleaningModal')" class="px-4 py-2 border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50">
                            Cancel
                        </button>
                        <button onclick="previewCleaning()" class="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700">
                            Preview Changes
                        </button>
                        <button onclick="applyCleaning()" class="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700">
                            Apply Cleaning
                        </button>
                    </div>
                </div>
            </div>
        </div>
    `;
    document.getElementById('modalContainer').innerHTML = modal;
}

// Show analysis interface
function showAnalysis() {
    // Implementation for analysis interface
    alert('Analysis interface coming soon!');
}

// Show visualization interface
function showVisualization() {
    // Implementation for visualization interface
    alert('Visualization interface coming soon!');
}

// Template imports
async function importTemplate(template) {
    closeModal('dataCollectionModal');
    
    // Show loading indicator
    showNotification('info', 'Importing data...', 'This may take a few moments');
    
    try {
        const response = await fetch(`${API_BASE_URL}/data/import-template`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ template })
        });
        
        if (response.ok) {
            const result = await response.json();
            showNotification('success', 'Import successful!', `Imported ${result.count} records`);
            updateRecentActivity();
        } else {
            throw new Error('Import failed');
        }
    } catch (error) {
        showNotification('error', 'Import failed', error.message);
    }
}

// Close modal
function closeModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.remove();
    }
}

// Show notification
function showNotification(type, title, message) {
    const colors = {
        success: 'green',
        error: 'red',
        info: 'blue',
        warning: 'yellow'
    };
    
    const color = colors[type] || 'gray';
    
    const notification = document.createElement('div');
    notification.className = `fixed top-4 right-4 p-4 bg-${color}-100 border border-${color}-400 text-${color}-700 rounded-lg shadow-lg z-50 max-w-md`;
    notification.innerHTML = `
        <div class="flex items-start">
            <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'} text-${color}-500 mr-3 text-xl"></i>
            <div class="flex-1">
                <p class="font-semibold">${title}</p>
                <p class="text-sm">${message}</p>
            </div>
            <button onclick="this.parentElement.parentElement.remove()" class="ml-4 text-${color}-500 hover:text-${color}-700">
                <i class="fas fa-times"></i>
            </button>
        </div>
    `;
    
    document.body.appendChild(notification);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (notification.parentElement) {
            notification.remove();
        }
    }, 5000);
}

// Update recent activity
function updateRecentActivity() {
    // Refresh the recent activity section
    location.reload();
}

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    // Check authentication
    checkAuth();
    
    // Load dashboard data
    loadDashboardData();
});

async function checkAuth() {
    // TODO: Implement authentication check
}

async function loadDashboardData() {
    // TODO: Load dashboard statistics and recent activity
}
