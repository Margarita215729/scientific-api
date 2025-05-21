// Константы и глобальные переменные
const API_BASE_URL = '';  // Empty string for relative paths
let mainChart = null;
let currentView = '3d';
let galaxyData = [];

// Цвета для каталогов
const CATALOG_COLORS = {
    'SDSS': '#1f77b4',
    'Euclid': '#ff7f0e',
    'DESI': '#2ca02c',
    'DES': '#d62728'
};

// Инициализация при загрузке страницы
document.addEventListener('DOMContentLoaded', () => {
    // Загрузка начальных данных
    loadCatalogStatus();
    loadStatistics();
    loadGalaxies();

    // Обработчик для формы фильтров
    document.getElementById('filter-form').addEventListener('submit', (e) => {
        e.preventDefault();
        loadGalaxies();
    });

    // Обработчики для кнопок переключения вида графика
    document.querySelectorAll('[data-view]').forEach(button => {
        button.addEventListener('click', (e) => {
            currentView = e.target.getAttribute('data-view');
            document.querySelectorAll('[data-view]').forEach(btn => {
                btn.classList.remove('active');
            });
            e.target.classList.add('active');
            updateVisualization();
        });
    });

    // Активируем первую кнопку по умолчанию
    document.querySelector('[data-view="3d"]').classList.add('active');
    
    // Инициализация формы поиска литературы ADS
    if (document.getElementById('ads-search-form')) {
        document.getElementById('ads-search-form').addEventListener('submit', (e) => {
            e.preventDefault();
            searchAdsLiterature();
        });
    }
});

// Загрузка статуса каталогов
async function loadCatalogStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/astro/status`);
        if (!response.ok) throw new Error('Не удалось загрузить статус каталогов');
        
        const data = await response.json();
        const statusContainer = document.getElementById('catalog-status');
        
        if (data.status === 'ok') {
            let html = '';
            data.catalogs.forEach(catalog => {
                const statusClass = catalog.available ? 'status-available' : 'status-unavailable';
                const statusText = catalog.available ? 'Доступен' : 'Недоступен';
                
                html += `
                <div class="catalog-item">
                    <span class="catalog-status ${statusClass}"></span>
                    <strong>${catalog.name}</strong>: 
                    <span class="ms-1">${statusText}</span>
                    ${catalog.available ? `<small class="text-muted ms-2">(${catalog.rows} объектов)</small>` : ''}
                </div>`;
            });
            statusContainer.innerHTML = html;
        } else {
            statusContainer.innerHTML = `<div class="alert alert-warning">Каталоги не загружены</div>`;
        }
    } catch (error) {
        console.error('Ошибка при загрузке статуса:', error);
        document.getElementById('catalog-status').innerHTML = `
            <div class="alert alert-danger">
                Ошибка при загрузке статуса каталогов: ${error.message}
            </div>`;
    }
}

// Загрузка общей статистики
async function loadStatistics() {
    try {
        const response = await fetch(`${API_BASE_URL}/astro/statistics`);
        if (!response.ok) throw new Error('Не удалось загрузить статистику');
        
        const data = await response.json();
        const statsContainer = document.getElementById('statistics');
        
        let html = `
            <div class="stat-item">
                <span class="stat-label">Всего галактик:</span>
                <span class="stat-value">${data.total_galaxies.toLocaleString()}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Красное смещение:</span>
                <span class="stat-value">${data.redshift.min} - ${data.redshift.max}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Среднее смещение:</span>
                <span class="stat-value">${data.redshift.mean.toFixed(3)}</span>
            </div>
            <hr>
            <h6>Источники данных:</h6>
        `;
        
        // Добавляем статистику по источникам
        Object.entries(data.sources).forEach(([source, count]) => {
            const colorClass = `color-${source.toLowerCase()}`;
            html += `
            <div class="stat-item">
                <span class="stat-label ${colorClass}">${source}:</span>
                <span class="stat-value">${count.toLocaleString()}</span>
            </div>`;
        });
        
        statsContainer.innerHTML = html;
    } catch (error) {
        console.error('Ошибка при загрузке статистики:', error);
        document.getElementById('statistics').innerHTML = `
            <div class="alert alert-danger">
                Ошибка при загрузке статистики: ${error.message}
            </div>`;
    }
}

// Загрузка данных галактик с применением фильтров
async function loadGalaxies() {
    try {
        // Получаем значения фильтров
        const source = document.getElementById('source').value;
        const minZ = document.getElementById('min-z').value;
        const maxZ = document.getElementById('max-z').value;
        const limit = document.getElementById('limit').value;
        
        // Формируем параметры запроса
        const params = new URLSearchParams();
        if (source) params.append('source', source);
        if (minZ) params.append('min_z', minZ);
        if (maxZ) params.append('max_z', maxZ);
        if (limit) params.append('limit', limit);
        
        // Показываем индикатор загрузки
        document.getElementById('galaxies-table').innerHTML = `
            <tr>
                <td colspan="7" class="text-center">
                    <div class="spinner-border text-primary" role="status">
                        <span class="visually-hidden">Загрузка...</span>
                    </div>
                </td>
            </tr>`;
        
        const response = await fetch(`${API_BASE_URL}/astro/galaxies?${params.toString()}`);
        if (!response.ok) throw new Error('Не удалось загрузить данные галактик');
        
        const data = await response.json();
        galaxyData = data.galaxies; // Сохраняем для визуализации
        
        // Обновляем таблицу
        updateGalaxyTable(galaxyData);
        
        // Обновляем визуализацию
        updateVisualization();
    } catch (error) {
        console.error('Ошибка при загрузке галактик:', error);
        document.getElementById('galaxies-table').innerHTML = `
            <tr>
                <td colspan="7" class="text-center text-danger">
                    Ошибка при загрузке данных: ${error.message}
                </td>
            </tr>`;
    }
}

// Обновление таблицы с данными галактик
function updateGalaxyTable(galaxies) {
    const tableBody = document.getElementById('galaxies-table');
    
    if (galaxies.length === 0) {
        tableBody.innerHTML = `
            <tr>
                <td colspan="7" class="text-center">
                    Нет данных, соответствующих фильтрам
                </td>
            </tr>`;
        return;
    }
    
    let html = '';
    galaxies.forEach(galaxy => {
        html += `
        <tr>
            <td>${galaxy.RA}</td>
            <td>${galaxy.DEC}</td>
            <td>${galaxy.redshift}</td>
            <td>${galaxy.source}</td>
            <td>${galaxy.X ? galaxy.X.toFixed(1) : 'N/A'}</td>
            <td>${galaxy.Y ? galaxy.Y.toFixed(1) : 'N/A'}</td>
            <td>${galaxy.Z ? galaxy.Z.toFixed(1) : 'N/A'}</td>
        </tr>`;
    });
    
    tableBody.innerHTML = html;
}

// Обновление визуализации данных
function updateVisualization() {
    if (galaxyData.length === 0) return;
    
    // Уничтожаем предыдущий график, если он существует
    if (mainChart) {
        mainChart.destroy();
    }
    
    if (currentView === '3d') {
        create3DVisualization();
    } else if (currentView === 'redshift') {
        createRedshiftVisualization();
    }
}

// Создание 3D визуализации распределения галактик
function create3DVisualization() {
    // Группируем данные по источникам
    const datasets = [];
    
    Object.keys(CATALOG_COLORS).forEach(source => {
        const filteredData = galaxyData.filter(galaxy => galaxy.source === source);
        
        if (filteredData.length > 0) {
            datasets.push({
                label: source,
                data: filteredData.map(galaxy => ({
                    x: galaxy.X,
                    y: galaxy.Y,
                    r: 5 + galaxy.redshift * 3 // Размер точки зависит от красного смещения
                })),
                backgroundColor: CATALOG_COLORS[source],
                borderColor: CATALOG_COLORS[source],
                borderWidth: 1
            });
        }
    });
    
    const ctx = document.getElementById('main-chart').getContext('2d');
    mainChart = new Chart(ctx, {
        type: 'bubble',
        data: {
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'X (Мпк)'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Y (Мпк)'
                    }
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const galaxy = galaxyData[context.dataIndex];
                            return [
                                `Источник: ${galaxy.source}`,
                                `Координаты: (${galaxy.X.toFixed(1)}, ${galaxy.Y.toFixed(1)}, ${galaxy.Z.toFixed(1)})`,
                                `Красное смещение: ${galaxy.redshift}`
                            ];
                        }
                    }
                },
                title: {
                    display: true,
                    text: '2D проекция распределения галактик в пространстве'
                }
            }
        }
    });
}

// Создание визуализации распределения красных смещений
function createRedshiftVisualization() {
    // Собираем данные по красным смещениям для каждого источника
    const datasets = [];
    const labels = [];
    
    // Создаем бины для гистограммы
    const minZ = Math.min(...galaxyData.map(g => g.redshift));
    const maxZ = Math.max(...galaxyData.map(g => g.redshift));
    const binSize = (maxZ - minZ) / 10;
    
    for (let i = 0; i < 10; i++) {
        const binStart = minZ + i * binSize;
        const binEnd = binStart + binSize;
        labels.push(`${binStart.toFixed(1)}-${binEnd.toFixed(1)}`);
    }
    
    // Создаем данные для каждого источника
    Object.keys(CATALOG_COLORS).forEach(source => {
        const sourceData = galaxyData.filter(galaxy => galaxy.source === source);
        
        if (sourceData.length > 0) {
            // Считаем количество галактик в каждом бине
            const counts = new Array(10).fill(0);
            
            sourceData.forEach(galaxy => {
                const binIndex = Math.min(9, Math.floor((galaxy.redshift - minZ) / binSize));
                counts[binIndex]++;
            });
            
            datasets.push({
                label: source,
                data: counts,
                backgroundColor: CATALOG_COLORS[source],
                borderColor: CATALOG_COLORS[source],
                borderWidth: 1
            });
        }
    });
    
    const ctx = document.getElementById('main-chart').getContext('2d');
    mainChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Красное смещение (z)'
                    },
                    stacked: true
                },
                y: {
                    title: {
                        display: true,
                        text: 'Количество галактик'
                    },
                    stacked: true
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Распределение галактик по красному смещению'
                }
            }
        }
    });
}

// Функции для работы с ADS API

// Поиск литературы через ADS API
async function searchAdsLiterature() {
    try {
        const searchType = document.getElementById('ads-search-type').value;
        const resultsContainer = document.getElementById('ads-results');
        
        resultsContainer.innerHTML = `
            <div class="d-flex justify-content-center my-5">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Загрузка...</span>
                </div>
            </div>`;
        
        let response;
        let data;
        
        switch (searchType) {
            case 'coordinates':
                const ra = document.getElementById('ads-ra').value;
                const dec = document.getElementById('ads-dec').value;
                const radius = document.getElementById('ads-radius').value || 0.1;
                
                if (!ra || !dec) {
                    resultsContainer.innerHTML = `
                        <div class="alert alert-warning">
                            Пожалуйста, укажите координаты (RA и DEC)
                        </div>`;
                    return;
                }
                
                response = await fetch(`${API_BASE_URL}/ads/search-by-coordinates?ra=${ra}&dec=${dec}&radius=${radius}`);
                break;
                
            case 'object':
                const objectName = document.getElementById('ads-object-name').value;
                
                if (!objectName) {
                    resultsContainer.innerHTML = `
                        <div class="alert alert-warning">
                            Пожалуйста, укажите название объекта
                        </div>`;
                    return;
                }
                
                response = await fetch(`${API_BASE_URL}/ads/search-by-object?object_name=${encodeURIComponent(objectName)}`);
                break;
                
            case 'catalog':
                const catalog = document.getElementById('ads-catalog').value;
                
                if (!catalog) {
                    resultsContainer.innerHTML = `
                        <div class="alert alert-warning">
                            Пожалуйста, выберите каталог
                        </div>`;
                    return;
                }
                
                response = await fetch(`${API_BASE_URL}/ads/search-by-catalog?catalog=${catalog}`);
                break;
                
            case 'lss':
                const keywords = document.getElementById('ads-keywords').value;
                const startYear = document.getElementById('ads-start-year').value || 2010;
                
                let url = `${API_BASE_URL}/ads/large-scale-structure?start_year=${startYear}`;
                if (keywords) {
                    const keywordArray = keywords.split(',').map(k => k.trim());
                    keywordArray.forEach(keyword => {
                        url += `&additional_keywords=${encodeURIComponent(keyword)}`;
                    });
                }
                
                response = await fetch(url);
                break;
                
            default:
                resultsContainer.innerHTML = `
                    <div class="alert alert-warning">
                        Выберите тип поиска
                    </div>`;
                return;
        }
        
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        
        data = await response.json();
        displayAdsResults(data, searchType, resultsContainer);
        
    } catch (error) {
        console.error('Ошибка при поиске литературы:', error);
        const resultsContainer = document.getElementById('ads-results');
        resultsContainer.innerHTML = `
            <div class="alert alert-danger">
                Ошибка при поиске: ${error.message}
            </div>`;
    }
}

// Отображение результатов поиска литературы
function displayAdsResults(data, searchType, container) {
    let html = '';
    let publications = [];
    let total = 0;
    
    // Извлекаем публикации в зависимости от типа поиска
    switch (searchType) {
        case 'coordinates':
        case 'object':
            publications = data.publications || [];
            total = data.count || 0;
            break;
        case 'catalog':
            publications = data.publications || [];
            total = data.total_found || 0;
            
            // Если есть статистика по ключевым словам, отображаем её
            if (data.keyword_stats) {
                html += `<div class="card mb-4">
                    <div class="card-header">
                        <h5 class="card-title">Ключевые слова для каталога ${data.catalog}</h5>
                    </div>
                    <div class="card-body">
                        <div class="row">`;
                        
                // Сортируем ключевые слова по частоте
                const sortedKeywords = Object.entries(data.keyword_stats)
                    .sort((a, b) => b[1] - a[1])
                    .slice(0, 20); // Показываем только топ-20
                
                sortedKeywords.forEach(([keyword, count]) => {
                    html += `<div class="col-md-6 mb-1">
                        <span class="badge bg-light text-dark">${keyword} (${count})</span>
                    </div>`;
                });
                
                html += `</div></div></div>`;
            }
            break;
        case 'lss':
            publications = data.publications || [];
            total = data.total_found || 0;
            
            // Отображаем статистику по годам
            if (data.year_stats) {
                html += `<div class="card mb-4">
                    <div class="card-header">
                        <h5 class="card-title">Публикации по годам</h5>
                    </div>
                    <div class="card-body">
                        <canvas id="years-chart" height="200"></canvas>
                    </div>
                </div>`;
            }
            break;
    }
    
    // Основная информация о результатах
    html += `<div class="alert alert-info">
        Найдено публикаций: ${total}. Показано: ${publications.length}.
    </div>`;
    
    // Таблица с публикациями
    if (publications.length > 0) {
        html += `<div class="table-responsive">
            <table class="table table-striped table-hover">
                <thead>
                    <tr>
                        <th>Название</th>
                        <th>Авторы</th>
                        <th>Год</th>
                        <th>Цитирования</th>
                        <th></th>
                    </tr>
                </thead>
                <tbody>`;
                
        publications.forEach(pub => {
            const title = Array.isArray(pub.title) ? pub.title[0] : pub.title || 'Без названия';
            const authors = Array.isArray(pub.author) ? pub.author.slice(0, 3).join(', ') + (pub.author.length > 3 ? ' и др.' : '') : 'Н/Д';
            const year = pub.year || 'Н/Д';
            const citations = pub.citation_count || 0;
            const bibcode = pub.bibcode || '';
            const doi = pub.doi ? (Array.isArray(pub.doi) ? pub.doi[0] : pub.doi) : '';
            
            html += `<tr>
                <td>${title}</td>
                <td>${authors}</td>
                <td>${year}</td>
                <td>${citations}</td>
                <td>`;
                
            if (bibcode) {
                html += `<a href="https://ui.adsabs.harvard.edu/abs/${bibcode}" target="_blank" class="btn btn-sm btn-outline-primary">ADS</a> `;
            }
            
            if (doi) {
                html += `<a href="https://doi.org/${doi}" target="_blank" class="btn btn-sm btn-outline-secondary">DOI</a>`;
            }
            
            html += `</td></tr>`;
        });
        
        html += `</tbody></table></div>`;
    } else {
        html += `<div class="alert alert-warning">
            Публикации не найдены. Попробуйте изменить параметры поиска.
        </div>`;
    }
    
    container.innerHTML = html;
    
    // Если у нас есть статистика по годам и мы ищем крупномасштабные структуры,
    // создаем график
    if (searchType === 'lss' && data.year_stats && document.getElementById('years-chart')) {
        const yearsData = Object.entries(data.year_stats)
            .sort((a, b) => parseInt(a[0]) - parseInt(b[0]));
        
        new Chart(document.getElementById('years-chart'), {
            type: 'bar',
            data: {
                labels: yearsData.map(item => item[0]),
                datasets: [{
                    label: 'Количество публикаций',
                    data: yearsData.map(item => item[1]),
                    backgroundColor: '#4e73df',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }
}

// Показать/скрыть поля в зависимости от типа поиска ADS
function updateAdsSearchFields() {
    const searchType = document.getElementById('ads-search-type').value;
    
    // Скрываем все поля
    document.querySelectorAll('.ads-search-field').forEach(field => {
        field.style.display = 'none';
    });
    
    // Показываем нужные поля в зависимости от типа поиска
    switch (searchType) {
        case 'coordinates':
            document.querySelectorAll('.field-coordinates').forEach(field => {
                field.style.display = 'block';
            });
            break;
        case 'object':
            document.querySelectorAll('.field-object').forEach(field => {
                field.style.display = 'block';
            });
            break;
        case 'catalog':
            document.querySelectorAll('.field-catalog').forEach(field => {
                field.style.display = 'block';
            });
            break;
        case 'lss':
            document.querySelectorAll('.field-lss').forEach(field => {
                field.style.display = 'block';
            });
            break;
    }
} 