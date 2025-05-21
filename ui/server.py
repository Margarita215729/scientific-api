"""
Простой сервер для обслуживания статических файлов пользовательского интерфейса.
"""
import http.server
import socketserver
import os
import sys
import webbrowser

# Порт по умолчанию
PORT = 8080

# Получаем абсолютный путь к директории UI
current_dir = os.path.dirname(os.path.abspath(__file__))

class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Обработчик HTTP-запросов с поддержкой CORS"""
    def end_headers(self):
        # Добавляем заголовки CORS для кросс-доменных запросов
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
        
    def do_OPTIONS(self):
        # Обработка предварительных запросов OPTIONS
        self.send_response(200)
        self.end_headers()

def run_server():
    """Запуск HTTP-сервера"""
    # Устанавливаем текущую директорию как рабочую для сервера
    os.chdir(current_dir)
    
    try:
        # Создаем сервер
        with socketserver.TCPServer(("", PORT), CORSHTTPRequestHandler) as httpd:
            print(f"Сервер запущен на порту {PORT}")
            print(f"Интерфейс доступен по адресу: http://localhost:{PORT}")
            print("Нажмите Ctrl+C для остановки сервера")
            
            # Открываем браузер
            webbrowser.open(f"http://localhost:{PORT}")
            
            # Запускаем сервер
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nСервер остановлен")
    except Exception as e:
        print(f"Ошибка: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Парсинг аргументов командной строки
    if len(sys.argv) > 1:
        try:
            PORT = int(sys.argv[1])
        except ValueError:
            print(f"Неверный формат порта: {sys.argv[1]}")
            print(f"Используется порт по умолчанию: {PORT}")
    
    run_server() 