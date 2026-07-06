/**
 * main.js - JavaScript for Rising Waters Flood Prediction
 * Handles form validation, interactivity, and UI enhancements
 */

// ============================================
// FORM VALIDATION
// ============================================

document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('predictionForm');
    
    if (form) {
        // Add validation on form submission
        form.addEventListener('submit', function(event) {
            let isValid = true;
            const inputs = form.querySelectorAll('.form-input');
            
            inputs.forEach(function(input) {
                const errorElement = document.getElementById(input.id + '-error');
                
                // Clear previous error
                if (errorElement) {
                    errorElement.textContent = '';
                }
                
                // Check if empty
                if (input.value.trim() === '') {
                    event.preventDefault();
                    isValid = false;
                    if (errorElement) {
                        errorElement.textContent = 'This field is required';
                    }
                    input.style.borderColor = '#e53e3e';
                    return;
                }
                
                // Check if valid number
                const numValue = parseFloat(input.value);
                if (isNaN(numValue)) {
                    event.preventDefault();
                    isValid = false;
                    if (errorElement) {
                        errorElement.textContent = 'Please enter a valid number';
                    }
                    input.style.borderColor = '#e53e3e';
                    return;
                }
                
                // Check if non-negative (for most weather parameters)
                if (numValue < 0) {
                    event.preventDefault();
                    isValid = false;
                    if (errorElement) {
                        errorElement.textContent = 'Value cannot be negative';
                    }
                    input.style.borderColor = '#e53e3e';
                    return;
                }
                
                input.style.borderColor = '#68d391';
            });
            
            if (!isValid) {
                event.preventDefault();
                // Scroll to first error
                const firstError = form.querySelector('.form-input[style*="border-color: #e53e3e"]');
                if (firstError) {
                    firstError.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    firstError.focus();
                }
            }
        });
        
        // Real-time validation on input
        const inputs = form.querySelectorAll('.form-input');
        inputs.forEach(function(input) {
            input.addEventListener('input', function() {
                const errorElement = document.getElementById(input.id + '-error');
                
                if (this.value.trim() === '') {
                    if (errorElement) {
                        errorElement.textContent = 'This field is required';
                    }
                    this.style.borderColor = '#e2e8f0';
                    return;
                }
                
                const numValue = parseFloat(this.value);
                if (isNaN(numValue) || numValue < 0) {
                    if (errorElement) {
                        errorElement.textContent = 'Please enter a valid non-negative number';
                    }
                    this.style.borderColor = '#fc8181';
                } else {
                    if (errorElement) {
                        errorElement.textContent = '';
                    }
                    this.style.borderColor = '#68d391';
                }
            });
            
            // Clear validation on focus
            input.addEventListener('focus', function() {
                const errorElement = document.getElementById(input.id + '-error');
                if (errorElement) {
                    errorElement.textContent = '';
                }
                this.style.borderColor = '#667eea';
            });
        });
    }
    
    // ============================================
    // SMOOTH SCROLLING FOR NAVIGATION LINKS
    // ============================================
    
    const navLinks = document.querySelectorAll('a[href^="#"]');
    navLinks.forEach(function(link) {
        link.addEventListener('click', function(e) {
            const targetId = this.getAttribute('href');
            if (targetId !== '#') {
                const targetElement = document.querySelector(targetId);
                if (targetElement) {
                    e.preventDefault();
                    targetElement.scrollIntoView({ behavior: 'smooth' });
                }
            }
        });
    });
    
    // ============================================
    // METER ANIMATION FOR RESULT PAGES
    // ============================================
    
    const meterFill = document.querySelector('.meter-fill');
    if (meterFill) {
        const targetWidth = meterFill.style.width;
        meterFill.style.width = '0%';
        setTimeout(function() {
            meterFill.style.width = targetWidth;
        }, 200);
    }
    
    // ============================================
    // RESET BUTTON FUNCTIONALITY
    // ============================================
    
    const resetButton = document.querySelector('button[type="reset"]');
    if (resetButton) {
        resetButton.addEventListener('click', function(e) {
            const inputs = document.querySelectorAll('.form-input');
            inputs.forEach(function(input) {
                input.value = '';
                input.style.borderColor = '#e2e8f0';
                const errorElement = document.getElementById(input.id + '-error');
                if (errorElement) {
                    errorElement.textContent = '';
                }
            });
            e.preventDefault();
        });
    }
    
    // ============================================
    // KEYBOARD SHORTCUTS
    // ============================================
    
    document.addEventListener('keydown', function(e) {
        // Press 'Ctrl+Enter' or 'Cmd+Enter' to submit form
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            const form = document.getElementById('predictionForm');
            if (form) {
                form.dispatchEvent(new Event('submit'));
            }
        }
    });
    
    // ============================================
    // CONSOLE WELCOME MESSAGE
    // ============================================
    
    console.log('%c🌊 Rising Waters - Flood Prediction System', 'font-size: 24px; font-weight: bold; color: #667eea;');
    console.log('%cPowered by XGBoost | Flask | AI', 'font-size: 14px; color: #718096;');
    console.log('%c📍 Visit: http://localhost:5000', 'font-size: 14px; color: #38a169;');
    
    console.log('📊 Available Features:');
    // This will be populated if feature names are available
});