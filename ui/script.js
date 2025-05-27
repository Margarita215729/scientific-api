// Basic frontend JavaScript for Scientific API

document.addEventListener('DOMContentLoaded', () => {
    // Подсветка активного пункта меню
    const navLinks = document.querySelectorAll('nav a');
    const currentPath = window.location.pathname;
    const adsPath = '/static/ads.html'; // или просто 'ads.html' если Vercel правильно настроит
    const astroPath = '/static/astro.html'; // или 'astro.html'

    navLinks.forEach(link => {
        const linkPath = link.getAttribute('href');
        if (linkPath === '/' && (currentPath === '/' || currentPath.endsWith('index.html'))) {
            link.classList.add('active');
        }
        // Для ads.html и astro.html, сравниваем концы путей, т.к. Vercel может добавлять /static/
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

    // Обработчик для формы поиска ADS на ads.html
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
            adsQueryInput.required = true; // По умолчанию основное поле запроса обязательно

            if (this.value === 'object') {
                objectParamsDiv.style.display = 'block';
                adsQueryInput.required = false; // Для поиска по объекту, имя объекта обязательно, а не общий запрос
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

            let apiPath = '/api/ads/search'; // Базовый путь для ADS поиска

            if (searchType === 'general') {
                params.append('query', query);
                params.append('search_type', 'general');
            } else if (searchType === 'object') {
                params.append('object_name', formData.get('object_name'));
                apiPath = '/api/ads/search-by-object'; // Специальный эндпоинт
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
                    throw new Error(`Ошибка API: ${response.status} ${response.statusText}. ${errorData.detail || ''}`);
                }
                const data = await response.json();
                displayAdsResults(data, resultsContainer);
            } catch (error) {
                resultsContainer.innerHTML = `<p class="error">Не удалось получить результаты: ${error.message}</p>`;
                console.error("ADS Search Error:", error);
            }
            loadingIndicator.style.display = 'none';
        });
    }

    // Обработчик для формы галактик на astro.html
    const galaxiesForm = document.getElementById('galaxiesForm');
    if (galaxiesForm) {
        galaxiesForm.addEventListener('submit', async function(event) {
            event.preventDefault();
            const formData = new FormData(this);
            const params = new URLSearchParams();
            for (const [key, value] of formData.entries()) {
                if (value) { // Добавляем параметр только если у него есть значение
                    params.append(key, value);
                }
            }

            const resultsContainer = document.getElementById('galaxiesTableContainer');
            const loadingIndicator = document.getElementById('loadingGalaxies');
            resultsContainer.innerHTML = '';
            loadingIndicator.style.display = 'block';

            try {
                const response = await fetch(`/api/astro/galaxies?${params.toString()}`);
                 if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(`Ошибка API: ${response.status} ${response.statusText}. ${errorData.detail || ''}`);
                }
                const data = await response.json();
                displayGalaxiesTable(data.galaxies, resultsContainer);
            } catch (error) {
                resultsContainer.innerHTML = `<p class="error">Не удалось получить данные галактик: ${error.message}</p>`;
                 console.error("Galaxies Fetch Error:", error);
            }
            loadingIndicator.style.display = 'none';
        });
    }
});

async function checkBackendStatus() {
    const resultContainer = document.getElementById('backend-status-result');
    if (!resultContainer) return;
    resultContainer.textContent = 'Проверка статуса...';
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        resultContainer.textContent = JSON.stringify(data, null, 2);
    } catch (error) {
        resultContainer.textContent = `Ошибка при проверке статуса бэкенда: ${error.message}`;
        console.error("Backend Status Error:", error);
    }
}

function displayAdsResults(data, container) {
    if (!data || (!data.publications && !data.results)) {
        container.innerHTML = '<p>Нет результатов для отображения.</p>';
        return;
    }

    const items = data.publications || data.results || [];
    if (items.length === 0) {
        container.innerHTML = '<p>Публикации не найдены.</p>';
        return;
    }

    const ul = document.createElement('ul');
    items.forEach(item => {
        const li = document.createElement('li');
        let title = Array.isArray(item.title) ? item.title.join(', ') : item.title;
        let authors = Array.isArray(item.author) ? item.author.join(', ') : (item.author || 'N/A');
        
        li.innerHTML = `
            <strong>${title || 'Без названия'}</strong><br>
            Авторы: ${authors}<br>
            Год: ${item.year || 'N/A'}<br>
            Bibcode: ${item.bibcode || 'N/A'}<br>
            DOI: ${item.doi ? `<a href="https://doi.org/${item.doi}" target="_blank">${item.doi}</a>` : 'N/A'}<br>
            Цитирования: ${item.citation_count !== undefined ? item.citation_count : 'N/A'}
            ${item.abstract ? `<p><em>Аннотация:</em> ${item.abstract.substring(0, 200)}...</p>` : ''}
        `;
        ul.appendChild(li);
    });
    container.appendChild(ul);
}

function displayGalaxiesTable(galaxies, container) {
    if (!galaxies || galaxies.length === 0) {
        container.innerHTML = '<p>Данные по галактикам не найдены.</p>';
        return;
    }

    const table = document.createElement('table');
    const thead = document.createElement('thead');
    const tbody = document.createElement('tbody');

    // Определяем заголовки на основе ключей первого объекта (если они есть)
    const headers = Object.keys(galaxies[0] || {});
    
    const headerRow = document.createElement('tr');
    headers.forEach(headerText => {
        const th = document.createElement('th');
        th.textContent = headerText;
        headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);

    galaxies.forEach(galaxy => {
        const row = document.createElement('tr');
        headers.forEach(header => {
            const cell = document.createElement('td');
            let value = galaxy[header];
            // Округляем числа для лучшего отображения
            if (typeof value === 'number') {
                value = parseFloat(value.toFixed(4));
            }
            cell.textContent = value !== null && value !== undefined ? value : 'N/A';
            row.appendChild(cell);
        });
        tbody.appendChild(row);
    });

    table.appendChild(thead);
    table.appendChild(tbody);
    container.appendChild(table);
}

async function getAstroStatus() {
    const resultContainer = document.getElementById('astro-status-result');
    if (!resultContainer) return;
    resultContainer.textContent = 'Получение статуса каталогов...';
    try {
        const response = await fetch('/api/astro/status');
        const data = await response.json();
        resultContainer.textContent = JSON.stringify(data, null, 2);
    } catch (error) {
        resultContainer.textContent = `Ошибка: ${error.message}`;
        console.error("Astro Status Error:", error);
    }
}

async function getAstroStats() {
    const resultContainer = document.getElementById('astro-stats-result');
    if (!resultContainer) return;
    resultContainer.textContent = 'Получение статистики каталогов...';
    try {
        const response = await fetch('/api/astro/statistics');
        const data = await response.json();
        resultContainer.textContent = JSON.stringify(data, null, 2);
    } catch (error) {
        resultContainer.textContent = `Ошибка: ${error.message}`;
        console.error("Astro Stats Error:", error);
    }
}

let downloadTaskId = null;
let downloadInterval = null;

async function downloadCatalogs() {
    const resultContainer = document.getElementById('download-status-result');
    const downloadBtn = document.getElementById('downloadBtn');
    if (!resultContainer || !downloadBtn) return;

    resultContainer.textContent = 'Запуск загрузки каталогов...';
    downloadBtn.disabled = true;
    downloadBtn.textContent = 'Загрузка...';

    try {
        const response = await fetch('/api/astro/download', { method: 'POST' });
        const data = await response.json();
        if (data.task_id) {
            downloadTaskId = data.task_id;
            resultContainer.textContent = `Задача загрузки запущена, ID: ${downloadTaskId}. Проверка статуса...`;
            // Запускаем интервальную проверку статуса
            if(downloadInterval) clearInterval(downloadInterval);
            downloadInterval = setInterval(checkDownloadStatus, 5000); 
        } else {
            resultContainer.textContent = `Ошибка запуска задачи: ${JSON.stringify(data)}`;
            downloadBtn.disabled = false;
            downloadBtn.textContent = 'Загрузить/обновить каталоги';
        }
    } catch (error) {
        resultContainer.textContent = `Ошибка при запуске загрузки: ${error.message}`;
        downloadBtn.disabled = false;
        downloadBtn.textContent = 'Загрузить/обновить каталоги';
        console.error("Download Catalogs Error:", error);
    }
}

async function checkDownloadStatus() {
    const resultContainer = document.getElementById('download-status-result');
    const downloadBtn = document.getElementById('downloadBtn');
    if (!downloadTaskId || !resultContainer || !downloadBtn) return;

    try {
        const response = await fetch(`/api/astro/download/${downloadTaskId}`);
        const data = await response.json();
        resultContainer.textContent = `Статус задачи ${downloadTaskId}: ${data.status} (${data.progress || 0}%) - ${data.message || ''}`;
        
        if (data.status === 'completed' || data.status === 'failed') {
            clearInterval(downloadInterval);
            downloadInterval = null;
            downloadTaskId = null;
            downloadBtn.disabled = false;
            downloadBtn.textContent = 'Загрузить/обновить каталоги';
            if(data.status === 'completed') {
                 resultContainer.textContent += '\nЗагрузка завершена! Можете обновить статус каталогов.';
            }
        }
    } catch (error) {
        resultContainer.textContent = `Ошибка проверки статуса задачи: ${error.message}`;
        clearInterval(downloadInterval);
        downloadInterval = null;
        downloadBtn.disabled = false;
        downloadBtn.textContent = 'Загрузить/обновить каталоги';
        console.error("Check Download Status Error:", error);
    }
} 