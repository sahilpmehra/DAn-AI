<script lang="ts">
    import Chart from 'chart.js/auto';

    type LineChartData = {
        labels: string[];
        values: number[];
        label: string;
    };

    let { data = {
        labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
        values: [65, 59, 80, 81, 56, 55],
        label: 'Monthly Sales'
    } } = $props<{ data?: LineChartData }>();

    let canvas: HTMLCanvasElement;
    let chart = $state<Chart | null>(null);

    $effect(() => {
        if (!canvas) return;

        const chartData = {
            labels: data.labels,
            datasets: [{
                label: data.label,
                data: data.values,
                fill: false,
                borderColor: 'rgb(75, 192, 192)',
                tension: 0.1
            }]
        };

        chart = new Chart(canvas, {
            type: 'line',
            data: chartData,
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: data.label
                    }
                }
            }
        });

        return () => {
            chart?.destroy();
        };
    });
</script>

<canvas bind:this={canvas}></canvas> 