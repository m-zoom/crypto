// Pattern Recognition System JavaScript

class PatternRecognitionApp {
    constructor() {
        this.isMonitoring = false;
        this.chart = null;
        this.pollingInterval = null;
        this.logs = [];
        
        this.initializeEventListeners();
        this.initializeChart();
        this.updateStatus('Ready to start monitoring', 'info');
    }

    initializeEventListeners() {
        document.getElementById('startBtn').addEventListener('click', () => this.startMonitoring());
        document.getElementById('stopBtn').addEventListener('click', () => this.stopMonitoring());
        document.getElementById('analyzeBtn').addEventListener('click', () => this.runHistoricalAnalysis());
        document.getElementById('clearLogsBtn').addEventListener('click', () => this.clearLogs());
    }

    initializeChart() {
        const ctx = document.getElementById('priceChart').getContext('2d');
        
        this.chart = new Chart(ctx, {
            type: 'candlestick',
            data: {
                datasets: [{
                    label: 'Price',
                    data: []
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    intersect: false,
                },
                scales: {
                    x: {
                        type: 'linear',
                        position: 'bottom',
                        title: {
                            display: true,
                            text: 'Candle Index'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Price'
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Candlestick Chart'
                    },
                    legend: {
                        display: false
                    }
                }
            }
        });
    }

    async startMonitoring() {
        const ticker = document.getElementById('tickerSelect').value;
        const timeframe = document.getElementById('timeframeSelect').value;

        try {
            this.setButtonState(true);
            this.updateStatus(`Starting monitoring for ${ticker} on ${timeframe}...`, 'warning');

            const response = await fetch('/start_monitoring', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ ticker, timeframe })
            });

            const result = await response.json();

            if (result.status === 'success') {
                this.isMonitoring = true;
                this.updateStatus(`Monitoring ${ticker} on ${timeframe} timeframe`, 'success');
                this.startPolling();
            } else {
                this.setButtonState(false);
                this.updateStatus(`Error: ${result.message}`, 'danger');
            }
        } catch (error) {
            this.setButtonState(false);
            this.updateStatus(`Network error: ${error.message}`, 'danger');
        }
    }

    async stopMonitoring() {
        try {
            const response = await fetch('/stop_monitoring', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });

            const result = await response.json();

            if (result.status === 'success') {
                this.isMonitoring = false;
                this.setButtonState(false);
                this.updateStatus('Monitoring stopped', 'secondary');
                this.stopPolling();
            } else {
                this.updateStatus(`Error stopping: ${result.message}`, 'danger');
            }
        } catch (error) {
            this.updateStatus(`Network error: ${error.message}`, 'danger');
        }
    }

    async runHistoricalAnalysis() {
        const ticker = document.getElementById('tickerSelect').value;
        const timeframe = document.getElementById('timeframeSelect').value;
        const analyzeBtn = document.getElementById('analyzeBtn');

        try {
            analyzeBtn.disabled = true;
            analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Analyzing...';
            this.updateStatus(`Running historical analysis for ${ticker}...`, 'info');

            const response = await fetch('/get_historical_analysis', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ ticker, timeframe })
            });

            const result = await response.json();

            if (result.status === 'success') {
                this.updatePatternResults(result.results);
                this.updateChart(result.chart_data);
                this.updateStatus(`Historical analysis completed for ${ticker}`, 'success');
            } else {
                this.updateStatus(`Analysis error: ${result.message}`, 'danger');
            }
        } catch (error) {
            this.updateStatus(`Network error: ${error.message}`, 'danger');
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = '<i class="fas fa-search me-1"></i>Analyze';
        }
    }

    startPolling() {
        this.pollingInterval = setInterval(() => {
            this.fetchMonitoringData();
        }, 5000); // Poll every 5 seconds
    }

    stopPolling() {
        if (this.pollingInterval) {
            clearInterval(this.pollingInterval);
            this.pollingInterval = null;
        }
    }

    async fetchMonitoringData() {
        if (!this.isMonitoring) return;

        try {
            const response = await fetch('/get_monitoring_data');
            const data = await response.json();

            if (!data.is_monitoring) {
                this.isMonitoring = false;
                this.setButtonState(false);
                this.updateStatus('Monitoring stopped by server', 'warning');
                this.stopPolling();
                return;
            }

            // Update UI with new data
            if (data.results) {
                this.updatePatternResults(data.results);
            }

            if (data.chart_data) {
                this.updateChart(data.chart_data);
            }

            if (data.logs && data.logs.length > this.logs.length) {
                this.updateLogs(data.logs);
            }

        } catch (error) {
            console.error('Polling error:', error);
        }
    }

    updatePatternResults(results) {
        const container = document.getElementById('patternResults');
        
        if (!results || Object.keys(results).length === 0) {
            container.innerHTML = `
                <div class="text-muted text-center">
                    <i class="fas fa-chart-area fa-2x mb-2"></i>
                    <p>No pattern detection results available.</p>
                </div>
            `;
            return;
        }

        let html = '';
        
        for (const [patternName, data] of Object.entries(results)) {
            if (data.error) {
                html += `
                    <div class="pattern-result error">
                        <div class="d-flex justify-content-between align-items-center mb-1">
                            <strong>${patternName}</strong>
                            <span class="badge bg-danger">Error</span>
                        </div>
                        <small class="text-muted">${data.error}</small>
                    </div>
                `;
            } else {
                const detected = data.detected;
                const confidence = data.confidence || 0;
                const status = detected ? 'detected' : 'not-detected';
                const badgeClass = detected ? 'bg-success' : 'bg-secondary';
                const badgeText = detected ? 'Detected' : 'Not Detected';
                const confidenceClass = confidence > 70 ? 'high' : confidence > 30 ? 'medium' : 'low';

                html += `
                    <div class="pattern-result ${status}">
                        <div class="d-flex justify-content-between align-items-center mb-1">
                            <strong>${patternName}</strong>
                            <span class="badge ${badgeClass}">${badgeText}</span>
                        </div>
                        <div class="d-flex justify-content-between align-items-center mb-1">
                            <small>Confidence: ${confidence.toFixed(1)}%</small>
                            <small class="text-muted">${data.model_type || 'Unknown'}</small>
                        </div>
                        <div class="confidence-bar">
                            <div class="confidence-fill ${confidenceClass}" style="width: ${confidence}%"></div>
                        </div>
                    </div>
                `;
            }
        }

        container.innerHTML = html;
    }

    updateChart(chartData) {
        if (!chartData || chartData.length === 0) return;

        this.chart.data.datasets[0].data = chartData;
        this.chart.update('none'); // Update without animation for real-time
    }

    updateLogs(logs) {
        this.logs = logs;
        const container = document.getElementById('logsContainer');

        if (logs.length === 0) {
            container.innerHTML = `
                <div class="text-muted text-center p-3">
                    <i class="fas fa-clipboard-list fa-2x mb-2"></i>
                    <p>No activity logs yet.</p>
                </div>
            `;
            return;
        }

        let html = '';
        logs.slice(-50).reverse().forEach(log => { // Show last 50 logs, newest first
            const icon = this.getLogIcon(log.type);
            html += `
                <div class="log-entry ${log.type}">
                    <div class="d-flex align-items-start">
                        <i class="${icon} me-2 mt-1"></i>
                        <div class="flex-grow-1">
                            <div class="d-flex justify-content-between">
                                <span class="fw-medium">${log.timestamp}</span>
                                ${log.pattern ? `<span class="badge bg-primary">${log.pattern}</span>` : ''}
                            </div>
                            <div class="mt-1">${log.message}</div>
                            ${log.confidence !== undefined ? `<small class="text-muted">Confidence: ${log.confidence.toFixed(1)}%</small>` : ''}
                        </div>
                    </div>
                </div>
            `;
        });

        container.innerHTML = html;
        container.scrollTop = 0; // Scroll to top (newest logs)
    }

    getLogIcon(type) {
        const icons = {
            'detected': 'fas fa-check-circle text-success',
            'not_detected': 'fas fa-times-circle text-secondary',
            'low_confidence': 'fas fa-exclamation-triangle text-warning',
            'error': 'fas fa-exclamation-circle text-danger',
            'info': 'fas fa-info-circle text-primary'
        };
        return icons[type] || 'fas fa-circle text-muted';
    }

    clearLogs() {
        this.logs = [];
        this.updateLogs([]);
    }

    setButtonState(monitoring) {
        document.getElementById('startBtn').disabled = monitoring;
        document.getElementById('stopBtn').disabled = !monitoring;
    }

    updateStatus(message, type) {
        const statusAlert = document.getElementById('statusAlert');
        const statusText = document.getElementById('statusText');
        
        // Update alert class
        statusAlert.className = `alert alert-${type} d-flex align-items-center`;
        
        // Update status indicator
        const indicator = statusAlert.querySelector('.status-indicator') || document.createElement('span');
        indicator.className = `status-indicator ${this.isMonitoring ? 'active' : 'inactive'}`;
        
        // Update text
        statusText.textContent = message;
        
        // Add indicator if it doesn't exist
        if (!statusAlert.querySelector('.status-indicator')) {
            statusAlert.insertBefore(indicator, statusAlert.firstChild);
        }
    }
}

// Initialize the application when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new PatternRecognitionApp();
});
