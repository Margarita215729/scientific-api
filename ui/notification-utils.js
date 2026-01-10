// Notification system for consistent user feedback
class NotificationManager {
    constructor() {
        this.notifications = new Map();
        this.createContainer();
    }

    createContainer() {
        if (!document.getElementById('notificationContainer')) {
            const container = document.createElement('div');
            container.id = 'notificationContainer';
            container.className = 'fixed top-4 right-4 z-50 space-y-3';
            container.setAttribute('aria-live', 'polite');
            container.setAttribute('aria-label', 'Notifications');
            document.body.appendChild(container);
        }
    }

    show(type, title, message, options = {}) {
        const id = Date.now().toString();
        const {
            duration = 5000,
            persistent = false,
            actions = []
        } = options;

        const typeConfig = {
            success: {
                bgColor: 'bg-green-50',
                borderColor: 'border-green-400',
                iconColor: 'text-green-400',
                titleColor: 'text-green-800',
                messageColor: 'text-green-700',
                icon: 'fas fa-check-circle'
            },
            error: {
                bgColor: 'bg-red-50',
                borderColor: 'border-red-400',
                iconColor: 'text-red-400',
                titleColor: 'text-red-800',
                messageColor: 'text-red-700',
                icon: 'fas fa-exclamation-circle'
            },
            warning: {
                bgColor: 'bg-yellow-50',
                borderColor: 'border-yellow-400',
                iconColor: 'text-yellow-400',
                titleColor: 'text-yellow-800',
                messageColor: 'text-yellow-700',
                icon: 'fas fa-exclamation-triangle'
            },
            info: {
                bgColor: 'bg-blue-50',
                borderColor: 'border-blue-400',
                iconColor: 'text-blue-400',
                titleColor: 'text-blue-800',
                messageColor: 'text-blue-700',
                icon: 'fas fa-info-circle'
            }
        };

        const config = typeConfig[type] || typeConfig.info;

        const notification = document.createElement('div');
        notification.id = `notification-${id}`;
        notification.className = `max-w-sm w-full ${config.bgColor} shadow-lg rounded-lg pointer-events-auto ring-1 ring-black ring-opacity-5 overflow-hidden transform transition-all duration-300 translate-x-full`;
        notification.setAttribute('role', 'alert');
        notification.setAttribute('aria-labelledby', `notification-title-${id}`);

        notification.innerHTML = `
            <div class="p-4">
                <div class="flex items-start">
                    <div class="flex-shrink-0">
                        <i class="${config.icon} ${config.iconColor}"></i>
                    </div>
                    <div class="ml-3 w-0 flex-1 pt-0.5">
                        <p id="notification-title-${id}" class="text-sm font-medium ${config.titleColor}">${title}</p>
                        <p class="mt-1 text-sm ${config.messageColor}">${message}</p>
                        ${actions.length > 0 ? `
                            <div class="mt-3 flex space-x-2">
                                ${actions.map(action => `
                                    <button onclick="${action.onclick}" 
                                            class="text-sm ${action.class || 'text-blue-600 hover:text-blue-500'}">
                                        ${action.text}
                                    </button>
                                `).join('')}
                            </div>
                        ` : ''}
                    </div>
                    ${!persistent ? `
                        <div class="ml-4 flex-shrink-0 flex">
                            <button onclick="notificationManager.remove('${id}')" 
                                    class="bg-white rounded-md inline-flex text-gray-400 hover:text-gray-500"
                                    aria-label="Close notification">
                                <i class="fas fa-times"></i>
                            </button>
                        </div>
                    ` : ''}
                </div>
            </div>
        `;

        const container = document.getElementById('notificationContainer');
        container.appendChild(notification);

        // Animate in
        setTimeout(() => {
            notification.classList.remove('translate-x-full');
        }, 50);

        this.notifications.set(id, notification);

        // Auto-remove if not persistent
        if (!persistent && duration > 0) {
            setTimeout(() => {
                this.remove(id);
            }, duration);
        }

        return id;
    }

    remove(id) {
        const notification = this.notifications.get(id);
        if (notification) {
            notification.classList.add('translate-x-full');
            setTimeout(() => {
                if (notification.parentElement) {
                    notification.remove();
                }
                this.notifications.delete(id);
            }, 300);
        }
    }

    clear() {
        this.notifications.forEach((_, id) => this.remove(id));
    }
}

// Global instance
const notificationManager = new NotificationManager();

// Backward compatibility
function showNotification(type, title, message, options) {
    return notificationManager.show(type, title, message, options);
}