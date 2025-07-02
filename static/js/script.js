class PatternRecognitionApp {
    constructor() {
        this.isMonitoring = false;
        this.pollingInterval = null;
        this.chart = null;
        this.initializeChart();
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        window.startMonitoring = () => this.startMonitoring();
        window.stopMonitoring = () => this.stopMonitoring();
    }

    initializeChart() {
        const ctx = document.getElementById('priceChart').getContext('2d');
        this.chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Price',
                    data: [],
                    borderColor: '#8B5CF6',
                    backgroundColor: 'rgba(139, 92, 246, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        grid: {
                            color: 'rgba(107, 70, 193, 0.2)'
                        },
                        ticks: {
                            color: '#CBD5E1'
                        }
                    },
                    y: {
                        grid: {
                            color: 'rgba(107, 70, 193, 0.2)'
                        },
                        ticks: {
                            color: '#CBD5E1'
                        }
                    }
                },
                elements: {
                    point: {
                        radius: 0,
                        hoverRadius: 6
                    }
                }
            }
        });
    }

    async startMonitoring() {
        const ticker = document.getElementById('ticker').value;
        const timeframe = document.getElementById('timeframe').value;

        if (this.isMonitoring) {
            return;
        }

        try {
            const response = await fetch('/start_monitoring', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ ticker, timeframe })
            });

            const result = await response.json();
            
            if (result.status === 'started') {
                this.isMonitoring = true;
                this.setButtonState(true);
                this.updateStatus('Monitoring started - analyzing patterns in real-time', 'monitoring');
                this.startPolling();
            } else {
                this.updateStatus('Failed to start monitoring', 'error');
            }
        } catch (error) {
            console.error('Error starting monitoring:', error);
            this.updateStatus('Error starting monitoring', 'error');
        }
    }

    async stopMonitoring() {
        if (!this.isMonitoring) {
            return;
        }

        try {
            const response = await fetch('/stop_monitoring', {
                method: 'POST'
            });

            const result = await response.json();
            
            if (result.status === 'stopped') {
                this.isMonitoring = false;
                this.setButtonState(false);
                this.updateStatus('Ready to start monitoring', 'ready');
                this.stopPolling();
            }
        } catch (error) {
            console.error('Error stopping monitoring:', error);
        }
    }

    startPolling() {
        this.pollingInterval = setInterval(() => {
            this.fetchMonitoringData();
        }, 5000);
    }

    stopPolling() {
        if (this.pollingInterval) {
            clearInterval(this.pollingInterval);
            this.pollingInterval = null;
        }
    }

    async fetchMonitoringData() {
        try {
            const response = await fetch('/get_monitoring_data');
            const data = await response.json();
            
            if (data.is_monitoring) {
                this.updatePatternResults(data.results);
                this.updateChart(data.chart_data);
            } else {
                // Monitoring stopped externally
                this.isMonitoring = false;
                this.setButtonState(false);
                this.updateStatus('Monitoring stopped', 'ready');
                this.stopPolling();
            }
        } catch (error) {
            console.error('Error fetching monitoring data:', error);
        }
    }

    updatePatternResults(results) {
        const container = document.getElementById('patternResults');
        
        if (!results || Object.keys(results).length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-chart-line"></i>
                    <div>Analyzing patterns... Please wait for results.</div>
                </div>
            `;
            return;
        }

        let html = '<div class="pattern-results">';
        
        for (const [patternName, result] of Object.entries(results)) {
            if (result.error) {
                html += `
                    <div class="pattern-item">
                        <div class="pattern-name">${this.formatPatternName(patternName)}</div>
                        <div class="pattern-status">
                            <span class="pattern-not-detected">Error: ${result.error}</span>
                        </div>
                    </div>
                `;
            } else {
                const isDetected = result.detected;
                const confidence = result.confidence ? result.confidence.toFixed(1) : '0.0';
                const modelType = result.model_type || 'Unknown';
                
                html += `
                    <div class="pattern-item">
                        <div>
                            <div class="pattern-name">${this.formatPatternName(patternName)}</div>
                            <small class="text-muted">${modelType}</small>
                        </div>
                        <div class="pattern-status">
                            <span class="${isDetected ? 'pattern-detected' : 'pattern-not-detected'}">
                                <i class="fas fa-${isDetected ? 'check-circle' : 'times-circle'} me-1"></i>
                                ${isDetected ? 'Detected' : 'Not Detected'}
                            </span>
                            <div class="confidence-badge">${confidence}%</div>
                        </div>
                    </div>
                `;
            }
        }
        
        html += '</div>';
        container.innerHTML = html;
    }

    updateChart(chartData) {
        if (!chartData || !Array.isArray(chartData) || chartData.length === 0) {
            return;
        }

        const labels = [];
        const prices = [];

        chartData.forEach((candle, index) => {
            labels.push(`Candle ${index + 1}`);
            prices.push(candle[3]); // Close price
        });

        this.chart.data.labels = labels;
        this.chart.data.datasets[0].data = prices;
        this.chart.update('none');
    }

    formatPatternName(name) {
        return name.replace(/_/g, ' ')
                   .replace(/\b\w/g, l => l.toUpperCase());
    }

    setButtonState(monitoring) {
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const ticker = document.getElementById('ticker');
        const timeframe = document.getElementById('timeframe');

        if (monitoring) {
            startBtn.classList.add('d-none');
            stopBtn.classList.remove('d-none');
            ticker.disabled = true;
            timeframe.disabled = true;
        } else {
            startBtn.classList.remove('d-none');
            stopBtn.classList.add('d-none');
            ticker.disabled = false;
            timeframe.disabled = false;
        }
    }

    updateStatus(message, type) {
        const indicator = document.getElementById('statusIndicator');
        indicator.className = `status-indicator ${type}`;
        
        let icon = 'fas fa-check-circle';
        if (type === 'monitoring') {
            icon = 'fas fa-chart-line';
        } else if (type === 'error') {
            icon = 'fas fa-exclamation-triangle';
        }
        
        indicator.innerHTML = `
            <i class="${icon}"></i>
            ${message}
        `;
    }
}

// Initialize the app when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new PatternRecognitionApp();
});