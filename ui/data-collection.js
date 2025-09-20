// Data collection and import functionality
class DataCollectionManager {
    constructor() {
        this.API_BASE_URL = '/api';
        this.setupEventListeners();
    }

    setupEventListeners() {
        // Clean up any existing listeners
        this.cleanup();
        
        // Store references for cleanup
        this.eventHandlers = new Map();
    }

    cleanup() {
        if (this.eventHandlers) {
            this.eventHandlers.forEach((handler, element) => {
                element.removeEventListener('change', handler);
            });
            this.eventHandlers.clear();
        }
    }

    showDataCollection() {
        const content = `
            <div class="space-y-4">
                <!-- Import Options -->
                <div class="grid grid-cols-2 gap-4">
                    <button onclick="dataCollectionManager.showAPIImport()" 
                            class="p-4 border-2 border-dashed border-gray-300 rounded-lg hover:border-indigo-500 hover:bg-indigo-50 transition-all"
                            aria-label="Import data from API">
                        <i class="fas fa-plug text-2xl text-indigo-600 mb-2"></i>
                        <p class="font-semibold">API Integration</p>
                        <p class="text-xs text-gray-600">NASA, ADS, SDSS, etc.</p>
                    </button>
                    
                    <button onclick="dataCollectionManager.showFileUpload()" 
                            class="p-4 border-2 border-dashed border-gray-300 rounded-lg hover:border-green-500 hover:bg-green-50 transition-all"
                            aria-label="Upload data files">
                        <i class="fas fa-file-upload text-2xl text-green-600 mb-2"></i>
                        <p class="font-semibold">File Upload</p>
                        <p class="text-xs text-gray-600">CSV, JSON, FITS, etc.</p>
                    </button>
                    
                    <button onclick="dataCollectionManager.showWebScraping()" 
                            class="p-4 border-2 border-dashed border-gray-300 rounded-lg hover:border-purple-500 hover:bg-purple-50 transition-all"
                            aria-label="Web scraping setup">
                        <i class="fas fa-spider text-2xl text-purple-600 mb-2"></i>
                        <p class="font-semibold">Web Scraping</p>
                        <p class="text-xs text-gray-600">Extract from websites</p>
                    </button>
                    
                    <button onclick="dataCollectionManager.showDatabaseConnect()" 
                            class="p-4 border-2 border-dashed border-gray-300 rounded-lg hover:border-yellow-500 hover:bg-yellow-50 transition-all"
                            aria-label="Connect to database">
                        <i class="fas fa-database text-2xl text-yellow-600 mb-2"></i>
                        <p class="font-semibold">Database</p>
                        <p class="text-xs text-gray-600">Connect to external DB</p>
                    </button>
                </div>
                
                <!-- Quick Start Templates -->
                <div class="mt-6">
                    <h4 class="text-sm font-semibold text-gray-700 mb-2">Quick Start Templates</h4>
                    <div class="space-y-2">
                        <button onclick="dataCollectionManager.importTemplate('sdss_galaxies')" 
                                class="w-full text-left p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-all">
                            <div class="flex justify-between items-center">
                                <div>
                                    <p class="font-medium text-gray-900">SDSS Galaxy Survey</p>
                                    <p class="text-sm text-gray-600">Pre-configured dataset with ~500k galaxies</p>
                                </div>
                                <i class="fas fa-arrow-right text-gray-400"></i>
                            </div>
                        </button>
                        
                        <button onclick="dataCollectionManager.importTemplate('ads_papers')" 
                                class="w-full text-left p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-all">
                            <div class="flex justify-between items-center">
                                <div>
                                    <p class="font-medium text-gray-900">ADS Publications</p>
                                    <p class="text-sm text-gray-600">Recent astrophysics papers from ADS</p>
                                </div>
                                <i class="fas fa-arrow-right text-gray-400"></i>
                            </div>
                        </button>
                        
                        <button onclick="dataCollectionManager.importTemplate('exoplanet_archive')" 
                                class="w-full text-left p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-all">
                            <div class="flex justify-between items-center">
                                <div>
                                    <p class="font-medium text-gray-900">Exoplanet Archive</p>
                                    <p class="text-sm text-gray-600">NASA's confirmed exoplanet catalog</p>
                                </div>
                                <i class="fas fa-arrow-right text-gray-400"></i>
                            </div>
                        </button>
                    </div>
                </div>
            </div>
        `;

        modalManager.showModal({
            id: 'dataCollectionModal',
            title: 'Import Data',
            content: content,
            size: 'medium'
        });
    }

    showAPIImport() {
        const content = `
            <form id="apiImportForm" onsubmit="dataCollectionManager.handleAPIImport(event)">
                <div class="space-y-4">
                    <div>
                        <label for="apiSource" class="block text-sm font-medium text-gray-700 mb-2">Data Source</label>
                        <select id="apiSource" name="source" class="w-full px-3 py-2 border border-gray-300 rounded-md" required>
                            <option value="">Select a data source...</option>
                            <option value="nasa_ads">NASA ADS</option>
                            <option value="arxiv">arXiv</option>
                            <option value="sdss">SDSS</option>
                            <option value="gaia">Gaia DR3</option>
                            <option value="simbad">SIMBAD</option>
                        </select>
                    </div>
                    
                    <div id="apiParameters">
                        <!-- Dynamic parameters will be loaded here -->
                    </div>
                    
                    <div id="apiTips" class="bg-blue-50 border border-blue-200 rounded-md p-3 text-sm text-blue-700">
                        Select a data source to see available parameters and tips.
                    </div>
                </div>
            </form>
        `;

        modalManager.showModal({
            id: 'apiImportModal',
            title: 'API Data Import',
            content: content,
            size: 'large',
            actions: [
                {
                    text: 'Cancel',
                    class: 'border border-gray-300 text-gray-700 hover:bg-gray-50',
                    onclick: "modalManager.closeModal('apiImportModal')"
                },
                {
                    text: 'Import Data',
                    class: 'bg-indigo-600 text-white hover:bg-indigo-700',
                    onclick: "dataCollectionManager.submitAPIImport()"
                }
            ]
        });

        // Setup dynamic parameter loading
        const sourceSelect = document.getElementById('apiSource');
        const handler = (event) => this.loadAPIParameters(event);
        sourceSelect.addEventListener('change', handler);
        this.eventHandlers.set(sourceSelect, handler);
    }

    loadAPIParameters(event) {
        const source = event.target.value;
        const parametersDiv = document.getElementById('apiParameters');
        const tipsDiv = document.getElementById('apiTips');
        
        if (!source) {
            parametersDiv.innerHTML = '';
            tipsDiv.innerHTML = 'Select a data source to see available parameters and tips.';
            return;
        }

        let parametersHTML = '';
        
        const parameterConfigs = {
            nasa_ads: {
                html: `
                    <div class="grid grid-cols-2 gap-4">
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-2">Search Query</label>
                            <input type="text" name="query" placeholder="e.g., galaxy formation" 
                                   class="w-full px-3 py-2 border border-gray-300 rounded-md" required>
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-2">Publication Year</label>
                            <input type="number" name="year" placeholder="2023" min="1900" max="2024"
                                   class="w-full px-3 py-2 border border-gray-300 rounded-md">
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-2">Maximum Results</label>
                            <input type="number" name="limit" value="100" min="1" max="2000"
                                   class="w-full px-3 py-2 border border-gray-300 rounded-md">
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-2">Sort By</label>
                            <select name="sort" class="w-full px-3 py-2 border border-gray-300 rounded-md">
                                <option value="date">Publication Date</option>
                                <option value="citation_count">Citation Count</option>
                                <option value="relevance">Relevance</option>
                            </select>
                        </div>
                    </div>
                `,
                tips: 'Use specific keywords for better results. You can search by author, title, or abstract content.'
            },
            arxiv: {
                html: `
                    <div class="grid grid-cols-2 gap-4">
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-2">Search Terms</label>
                            <input type="text" name="search_query" placeholder="e.g., machine learning" 
                                   class="w-full px-3 py-2 border border-gray-300 rounded-md" required>
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-2">Category</label>
                            <select name="category" class="w-full px-3 py-2 border border-gray-300 rounded-md">
                                <option value="cat:astro-ph">Astrophysics</option>
                                <option value="cat:physics">Physics</option>
                                <option value="cat:math">Mathematics</option>
                                <option value="cat:cs">Computer Science</option>
                                <option value="cat:q-bio">Quantitative Biology</option>
                            </select>
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-2">Maximum Results</label>
                            <input type="number" name="max_results" value="100" min="1" max="2000"
                                   class="w-full px-3 py-2 border border-gray-300 rounded-md">
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-2">Sort By</label>
                            <select name="sortBy" class="w-full px-3 py-2 border border-gray-300 rounded-md">
                                <option value="submittedDate">Submission Date</option>
                                <option value="lastUpdatedDate">Last Updated</option>
                                <option value="relevance">Relevance</option>
                            </select>
                        </div>
                    </div>
                `,
                tips: 'arXiv searches work best with specific scientific terms. Use category filters to narrow down results.'
            }
        };

        const config = parameterConfigs[source];
        if (config) {
            parametersDiv.innerHTML = config.html;
            tipsDiv.innerHTML = config.tips;
        }
    }

    async importTemplate(template) {
        modalManager.closeModal('dataCollectionModal');
        
        const notificationId = notificationManager.show(
            'info', 
            'Importing data...', 
            'This may take a few moments'
        );
        
        try {
            const response = await fetch(`${this.API_BASE_URL}/data/import-template`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ template })
            });
            
            notificationManager.remove(notificationId);
            
            if (response.ok) {
                const result = await response.json();
                notificationManager.show(
                    'success', 
                    'Import successful!', 
                    `Imported ${result.count} records`
                );
                this.updateRecentActivity();
            } else {
                throw new Error('Import failed');
            }
        } catch (error) {
            notificationManager.remove(notificationId);
            notificationManager.show('error', 'Import failed', error.message);
        }
    }

    submitAPIImport() {
        const form = document.getElementById('apiImportForm');
        if (form && form.checkValidity()) {
            // Handle form submission
            const formData = new FormData(form);
            console.log('API Import data:', Object.fromEntries(formData));
            modalManager.closeModal('apiImportModal');
            notificationManager.show('info', 'API import started', 'Processing your request...');
        }
    }

    showFileUpload() {
        // Implementation for file upload modal
        modalManager.showModal({
            id: 'fileUploadModal',
            title: 'Upload Data Files',
            content: '<p>File upload interface coming soon...</p>',
            size: 'medium'
        });
    }

    showWebScraping() {
        modalManager.showModal({
            id: 'webScrapingModal',
            title: 'Web Scraping Setup',
            content: '<p>Web scraping interface coming soon...</p>',
            size: 'medium'
        });
    }

    showDatabaseConnect() {
        modalManager.showModal({
            id: 'databaseModal',
            title: 'Database Connection',
            content: '<p>Database connection interface coming soon...</p>',
            size: 'medium'
        });
    }

    updateRecentActivity() {
        // More efficient update without page reload
        const activitySection = document.querySelector('[data-section="recent-activity"]');
        if (activitySection) {
            // Update the activity section content
            console.log('Updating recent activity...');
        }
    }
}

// Global instance
const dataCollectionManager = new DataCollectionManager();