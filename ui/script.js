// Frontend JavaScript for Scientific API

document.addEventListener('DOMContentLoaded', () => {
    // Highlight active menu item
    const navLinks = document.querySelectorAll('nav a');
    const currentPath = window.location.pathname;
    const adsPath = '/static/ads.html';
    const astroPath = '/static/astro.html';

    navLinks.forEach(link => {
        const linkPath = link.getAttribute('href');
        if (linkPath === '/' && (currentPath === '/' || currentPath.endsWith('index.html'))) {
            link.classList.add('active');
        }
        else if (linkPath.endsWith('ads.html') && currentPath.endsWith(adsPath)) {
            link.classList.add('active');
        }
        else if (linkPath.endsWith('astro.html') && currentPath.endsWith(astroPath)) {
            link.classList.add('active');
        }
         else if (linkPath === '/docs' && currentPath.startsWith('/docs')) {
            link.classList.add('active');
        }
    });

    // ADS search form handler
    const searchFormAds = document.getElementById('searchFormAds');
    if (searchFormAds) {
        const searchTypeSelect = document.getElementById('searchType');
        const objectParamsDiv = document.getElementById('objectParams');
        const coordsParamsDiv = document.getElementById('coordsParams');
        const catalogParamsDiv = document.getElementById('catalogParams');
        const adsQueryInput = document.getElementById('adsQuery');

        searchTypeSelect.addEventListener('change', function() {
            objectParamsDiv.style.display = 'none';
            coordsParamsDiv.style.display = 'none';
            catalogParamsDiv.style.display = 'none';
            adsQueryInput.required = true;

            if (this.value === 'object') {
                objectParamsDiv.style.display = 'block';
                adsQueryInput.required = false;
            } else if (this.value === 'coordinates') {
                coordsParamsDiv.style.display = 'block';
                adsQueryInput.required = false;
            } else if (this.value === 'catalog') {
                catalogParamsDiv.style.display = 'block';
                adsQueryInput.required = false;
            }
        });

        searchFormAds.addEventListener('submit', async function(event) {
            event.preventDefault();
            const formData = new FormData(this);
            const params = new URLSearchParams();
            
            const query = formData.get('query');
            const searchType = formData.get('search_type');
            params.append('max_results', formData.get('max_results'));

            let apiPath = '/api/ads/search';

            if (searchType === 'general') {
                params.append('query', query);
                params.append('search_type', 'general');
            } else if (searchType === 'object') {
                params.append('object_name', formData.get('object_name'));
                apiPath = '/api/ads/search-by-object';
            } else if (searchType === 'coordinates') {
                params.append('ra', formData.get('ra'));
                params.append('dec', formData.get('dec'));
                params.append('radius', formData.get('radius'));
                apiPath = '/api/ads/search-by-coordinates';
            } else if (searchType === 'catalog') {
                params.append('catalog', formData.get('catalog_name'));
                apiPath = '/api/ads/search-by-catalog';
            }

            const resultsContainer = document.getElementById('adsResultsContainer');
            const loadingIndicator = document.getElementById('loadingAds');
            resultsContainer.innerHTML = '';
            loadingIndicator.style.display = 'block';

            try {
                const response = await fetch(`${apiPath}?${params.toString()}`);
                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(`API Error: ${response.status} ${response.statusText}. ${errorData.detail || ''}`);
                }
                const data = await response.json();
                displayAdsResults(data, resultsContainer);
            } catch (error) {
                resultsContainer.innerHTML = `<p class="error">Failed to get results: ${error.message}</p>`;
                console.error("ADS Search Error:", error);
            }
            loadingIndicator.style.display = 'none';
        });
    }
});

// Backend status check
async function checkBackendStatus() {
    const resultElement = document.getElementById('backend-status-result');
    resultElement.textContent = 'Checking backend status...';
    
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        resultElement.textContent = JSON.stringify(data, null, 2);
    } catch (error) {
        resultElement.textContent = `Error: ${error.message}`;
    }
}

// Display ADS search results
function displayAdsResults(data, container) {
    if (!data.publications || data.publications.length === 0) {
        container.innerHTML = '<p>No publications found.</p>';
        return;
    }

    let html = `<h3>Found ${data.count} publications:</h3>`;
    
    data.publications.forEach((pub, index) => {
        html += `
            <div class="publication-item">
                <h4>${index + 1}. ${pub.title || 'No title'}</h4>
                <p><strong>Authors:</strong> ${pub.author ? pub.author.join(', ') : 'Unknown'}</p>
                <p><strong>Year:</strong> ${pub.year || 'Unknown'}</p>
                <p><strong>Citations:</strong> ${pub.citation_count || 0}</p>
                ${pub.doi ? `<p><strong>DOI:</strong> <a href="https://doi.org/${pub.doi}" target="_blank">${pub.doi}</a></p>` : ''}
                ${pub.abstract ? `<p><strong>Abstract:</strong> ${pub.abstract.substring(0, 200)}...</p>` : ''}
            </div>
        `;
    });
    
    container.innerHTML = html;
}

// Astronomical data functions
async function loadCatalogData(catalogName) {
    const container = document.getElementById('resultsContainer');
    const loadingIndicator = document.getElementById('loadingIndicator');
    
    container.innerHTML = '';
    loadingIndicator.style.display = 'block';
    
    try {
        const response = await fetch(`/api/astro/catalog/${catalogName}`);
        const data = await response.json();
        
        displayCatalogData(data, container);
    } catch (error) {
        container.innerHTML = `<p class="error">Error loading catalog data: ${error.message}</p>`;
    }
    
    loadingIndicator.style.display = 'none';
}

function displayCatalogData(data, container) {
    let html = `<h3>${data.catalog_name} Data</h3>`;
    
    if (data.statistics) {
        html += '<h4>Statistics:</h4>';
        html += '<ul>';
        for (const [key, value] of Object.entries(data.statistics)) {
            html += `<li><strong>${key}:</strong> ${value}</li>`;
        }
        html += '</ul>';
    }
    
    if (data.objects && data.objects.length > 0) {
        html += `<h4>Sample Objects (${data.objects.length}):</h4>`;
        html += '<table class="data-table">';
        html += '<thead><tr>';
        
        // Get column headers from first object
        const headers = Object.keys(data.objects[0]);
        headers.forEach(header => {
            html += `<th>${header}</th>`;
        });
        html += '</tr></thead><tbody>';
        
        // Add data rows
        data.objects.forEach(obj => {
            html += '<tr>';
            headers.forEach(header => {
                html += `<td>${obj[header] || ''}</td>`;
            });
            html += '</tr>';
        });
        
        html += '</tbody></table>';
    }
    
    container.innerHTML = html;
}

async function getStatistics() {
    const container = document.getElementById('resultsContainer');
    const loadingIndicator = document.getElementById('loadingIndicator');
    
    container.innerHTML = '';
    loadingIndicator.style.display = 'block';
    
    try {
        const response = await fetch('/api/astro/statistics');
        const data = await response.json();
        
        let html = '<h3>Catalog Statistics</h3>';
        html += '<div class="stats-grid">';
        
        for (const [catalog, stats] of Object.entries(data.statistics)) {
            html += `
                <div class="stat-card">
                    <h4>${catalog}</h4>
                    <ul>
                        <li><strong>Total Objects:</strong> ${stats.total_objects || 0}</li>
                        <li><strong>Redshift Range:</strong> ${stats.redshift_range || 'N/A'}</li>
                        <li><strong>Magnitude Range:</strong> ${stats.magnitude_range || 'N/A'}</li>
                    </ul>
                </div>
            `;
        }
        
        html += '</div>';
        container.innerHTML = html;
        
    } catch (error) {
        container.innerHTML = `<p class="error">Error getting statistics: ${error.message}</p>`;
    }
    
    loadingIndicator.style.display = 'none';
}

function showFilterForm() {
    const container = document.getElementById('resultsContainer');
    
    const html = `
        <h3>Filter Galaxies</h3>
        <form id="filterForm">
            <div class="form-group">
                <label for="catalogSource">Catalog Source:</label>
                <select id="catalogSource" name="catalog_source">
                    <option value="all">All Catalogs</option>
                    <option value="SDSS">SDSS</option>
                    <option value="DESI">DESI</option>
                    <option value="DES">DES</option>
                    <option value="Euclid">Euclid</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="minRedshift">Min Redshift:</label>
                <input type="number" id="minRedshift" name="min_redshift" step="0.001">
            </div>
            
            <div class="form-group">
                <label for="maxRedshift">Max Redshift:</label>
                <input type="number" id="maxRedshift" name="max_redshift" step="0.001">
            </div>
            
            <div class="form-group">
                <label for="minMagnitude">Min Magnitude:</label>
                <input type="number" id="minMagnitude" name="min_magnitude" step="0.1">
            </div>
            
            <div class="form-group">
                <label for="maxMagnitude">Max Magnitude:</label>
                <input type="number" id="maxMagnitude" name="max_magnitude" step="0.1">
            </div>
            
            <div class="form-group">
                <label for="limit">Limit:</label>
                <input type="number" id="limit" name="limit" value="100" min="1" max="1000">
            </div>
            
            <button type="submit">Filter Galaxies</button>
        </form>
    `;
    
    container.innerHTML = html;
    
    // Add form handler
    const filterForm = document.getElementById('filterForm');
    filterForm.addEventListener('submit', async function(event) {
        event.preventDefault();
        await filterGalaxies();
    });
}

async function filterGalaxies() {
    const form = document.getElementById('filterForm');
    const formData = new FormData(form);
    const params = new URLSearchParams();
    
    for (const [key, value] of formData.entries()) {
        if (value) {
            params.append(key, value);
        }
    }
    
    const container = document.getElementById('resultsContainer');
    const loadingIndicator = document.getElementById('loadingIndicator');
    
    loadingIndicator.style.display = 'block';
    
    try {
        const response = await fetch(`/api/astro/filter?${params.toString()}`);
        const data = await response.json();
        
        displayFilteredGalaxies(data, container);
    } catch (error) {
        container.innerHTML = `<p class="error">Error filtering galaxies: ${error.message}</p>`;
    }
    
    loadingIndicator.style.display = 'none';
}

function displayFilteredGalaxies(data, container) {
    let html = `<h3>Filtered Galaxies (${data.count} found)</h3>`;
    
    if (data.galaxies && data.galaxies.length > 0) {
        html += '<table class="data-table">';
        html += '<thead><tr>';
        
        const headers = Object.keys(data.galaxies[0]);
        headers.forEach(header => {
            html += `<th>${header}</th>`;
        });
        html += '</tr></thead><tbody>';
        
        data.galaxies.forEach(galaxy => {
            html += '<tr>';
            headers.forEach(header => {
                html += `<td>${galaxy[header] || ''}</td>`;
            });
            html += '</tr>';
        });
        
        html += '</tbody></table>';
    } else {
        html += '<p>No galaxies found matching the criteria.</p>';
    }
    
    container.innerHTML = html;
}

async function prepareMLDataset() {
    const container = document.getElementById('resultsContainer');
    const loadingIndicator = document.getElementById('loadingIndicator');
    
    container.innerHTML = '';
    loadingIndicator.style.display = 'block';
    
    try {
        const response = await fetch('/api/ml/prepare-dataset', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                target_variable: 'redshift',
                test_size: 0.2
            })
        });
        
        const data = await response.json();
        
        let html = '<h3>ML Dataset Preparation</h3>';
        html += `<p><strong>Status:</strong> Dataset prepared successfully</p>`;
        html += `<p><strong>Training Samples:</strong> ${data.train_samples}</p>`;
        html += `<p><strong>Test Samples:</strong> ${data.test_samples}</p>`;
        html += `<p><strong>Features:</strong> ${data.features.join(', ')}</p>`;
        html += `<p><strong>Target Variable:</strong> ${data.target_variable}</p>`;
        
        html += '<h4>Dataset Information:</h4>';
        html += `<p><strong>Total Objects:</strong> ${data.dataset_info.total_objects}</p>`;
        html += `<p><strong>Catalog Sources:</strong> ${data.dataset_info.catalog_sources.join(', ')}</p>`;
        html += `<p><strong>Feature Count:</strong> ${data.dataset_info.feature_count}</p>`;
        
        container.innerHTML = html;
        
    } catch (error) {
        container.innerHTML = `<p class="error">Error preparing ML dataset: ${error.message}</p>`;
    }
    
    loadingIndicator.style.display = 'none';
} 