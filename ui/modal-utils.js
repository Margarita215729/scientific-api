// Modal utilities for consistent interface
class ModalManager {
    constructor() {
        this.activeModals = new Set();
        this.bindGlobalEvents();
    }

    bindGlobalEvents() {
        // ESC key to close modals
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.activeModals.size > 0) {
                this.closeTopModal();
            }
        });

        // Click outside to close
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('modal-overlay')) {
                this.closeModal(e.target.querySelector('.modal-content')?.dataset.modalId);
            }
        });
    }

    createModal(config) {
        const {
            id,
            title,
            content,
            actions = [],
            size = 'medium',
            closeButton = true
        } = config;

        const sizeClasses = {
            small: 'w-11/12 md:w-1/3',
            medium: 'w-11/12 md:w-3/4 lg:w-1/2', 
            large: 'w-11/12 md:w-3/4 lg:w-2/3',
            xlarge: 'w-11/12 md:w-5/6'
        };

        const modal = `
            <div class="modal-overlay fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full z-50" 
                 id="${id}" 
                 role="dialog" 
                 aria-modal="true" 
                 aria-labelledby="${id}-title">
                <div class="modal-content relative top-20 mx-auto p-5 border ${sizeClasses[size]} shadow-lg rounded-md bg-white"
                     data-modal-id="${id}">
                    <div class="flex justify-between items-center mb-4">
                        <h3 id="${id}-title" class="text-xl font-bold text-gray-900">${title}</h3>
                        ${closeButton ? `
                            <button onclick="modalManager.closeModal('${id}')" 
                                    class="text-gray-400 hover:text-gray-600"
                                    aria-label="Close modal">
                                <i class="fas fa-times"></i>
                            </button>
                        ` : ''}
                    </div>
                    
                    <div class="modal-body">
                        ${content}
                    </div>
                    
                    ${actions.length > 0 ? `
                        <div class="flex justify-end space-x-3 mt-6">
                            ${actions.map(action => `
                                <button onclick="${action.onclick}" 
                                        class="px-4 py-2 ${action.class || 'bg-gray-300 text-gray-700'} rounded-md hover:opacity-90">
                                    ${action.text}
                                </button>
                            `).join('')}
                        </div>
                    ` : ''}
                </div>
            </div>
        `;

        return modal;
    }

    showModal(config) {
        const modal = this.createModal(config);
        const container = document.getElementById('modalContainer');
        if (container) {
            container.innerHTML = modal;
            this.activeModals.add(config.id);
            
            // Focus management
            setTimeout(() => {
                const modalElement = document.getElementById(config.id);
                const firstFocusable = modalElement.querySelector('button, input, select, textarea, [tabindex]:not([tabindex="-1"])');
                if (firstFocusable) {
                    firstFocusable.focus();
                }
            }, 100);
        }
    }

    closeModal(modalId) {
        const modal = document.getElementById(modalId);
        if (modal) {
            modal.remove();
            this.activeModals.delete(modalId);
        }
    }

    closeTopModal() {
        if (this.activeModals.size > 0) {
            const topModal = Array.from(this.activeModals).pop();
            this.closeModal(topModal);
        }
    }
}

// Global instance
const modalManager = new ModalManager();

// Backward compatibility
function closeModal(modalId) {
    modalManager.closeModal(modalId);
}