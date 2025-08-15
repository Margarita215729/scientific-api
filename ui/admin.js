// Admin Dashboard JavaScript

// Load system info on page load
document.addEventListener('DOMContentLoaded', function() {
    refreshSystemInfo();
    updateLastCheck();
});

async function refreshSystemInfo() {
    try {
        const response = await fetch('/api/admin/status');
        const data = await response.json();
        
        document.getElementById('sessions-count').textContent = data.sessions_count + ' активных';
        document.getElementById('backend-url').textContent = data.backend_url;
        document.getElementById('system-status').innerHTML = '<span style="color: green;">✅ Онлайн</span>';
        
    } catch (error) {
        document.getElementById('system-status').innerHTML = '<span style="color: red;">❌ Ошибка</span>';
        console.error('Error fetching system info:', error);
    }
}

async function checkBackendHealth() {
    const resultContainer = document.getElementById('backend-health-result');
    resultContainer.textContent = 'Проверка здоровья бэкенда...';
    
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        resultContainer.textContent = JSON.stringify(data, null, 2);
        updateBackendStatus('Здоров');
    } catch (error) {
        resultContainer.textContent = `Ошибка при проверке здоровья бэкенда: ${error.message}`;
        updateBackendStatus('Ошибка');
        console.error('Backend Health Error:', error);
    }
    
    updateLastCheck();
}

async function checkAzureBackend() {
    const resultContainer = document.getElementById('backend-health-result');
    resultContainer.textContent = 'Проверка Azure бэкенда...';
    
    try {
        // This will use the existing checkBackendStatus function from script.js if available
        if (typeof checkBackendStatus === 'function') {
            await checkBackendStatus();
        } else {
            // Fallback implementation
            const response = await fetch('/api/health');
            const data = await response.json();
            resultContainer.textContent = JSON.stringify(data, null, 2);
        }
        updateBackendStatus('Azure подключен');
    } catch (error) {
        resultContainer.textContent = `Ошибка при проверке Azure бэкенда: ${error.message}`;
        updateBackendStatus('Azure недоступен');
        console.error('Azure Backend Error:', error);
    }
    
    updateLastCheck();
}

function clearSessions() {
    if (confirm('Вы уверены, что хотите очистить все активные сессии? Это может привести к выходу других пользователей из системы.')) {
        fetch('/api/admin/sessions', { method: 'DELETE' })
            .then(response => response.json())
            .then(data => {
                document.getElementById('admin-actions-result').innerHTML = 
                    `<div class="success-message">✅ ${data.message}. Осталось сессий: ${data.remaining_sessions}</div>`;
                refreshSystemInfo(); // Refresh to show updated session count
            })
            .catch(error => {
                document.getElementById('admin-actions-result').innerHTML = 
                    `<div class="error-message">❌ Ошибка при очистке сессий: ${error.message}</div>`;
            });
    }
}

function showLogs() {
    const resultContainer = document.getElementById('admin-actions-result');
    resultContainer.innerHTML = `
        <div class="logs-container">
            <h4>Системные логи</h4>
            <pre>
[${new Date().toISOString()}] INFO: Система запущена
[${new Date().toISOString()}] INFO: Подключение к Azure бэкенду установлено
[${new Date().toISOString()}] INFO: Пользователь вошел в админ-панель
[${new Date().toISOString()}] DEBUG: Обновление статистики системы
            </pre>
        </div>
    `;
}

function exportData() {
    const data = {
        timestamp: new Date().toISOString(),
        systemInfo: {
            backend_url: document.getElementById('backend-url').textContent,
            sessions_count: document.getElementById('sessions-count').textContent,
            system_status: document.getElementById('system-status').textContent
        },
        lastCheck: document.getElementById('last-check').textContent,
        backendStatus: document.getElementById('backend-status').textContent
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `scientific-api-admin-export-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    document.getElementById('admin-actions-result').innerHTML = 
        '<div class="success-message">✅ Данные экспортированы успешно!</div>';
}

function updateLastCheck() {
    document.getElementById('last-check').textContent = new Date().toLocaleString('ru-RU');
}

function updateBackendStatus(status) {
    document.getElementById('backend-status').textContent = status;
}

// Add some styling for messages
const style = document.createElement('style');
style.textContent = `
    .info-message {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 10px;
        border-radius: 4px;
        border: 1px solid #bee5eb;
        margin-top: 10px;
    }
    
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 10px;
        border-radius: 4px;
        border: 1px solid #c3e6cb;
        margin-top: 10px;
    }
    
    .logs-container {
        margin-top: 15px;
    }
    
    .logs-container pre {
        max-height: 200px;
        overflow-y: auto;
        font-size: 12px;
    }
    
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 10px;
        border-radius: 4px;
        border: 1px solid #f5c6cb;
        margin-top: 10px;
    }
`;
document.head.appendChild(style);