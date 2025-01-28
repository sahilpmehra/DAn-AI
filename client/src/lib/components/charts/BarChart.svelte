<script lang="ts">
    import Chart from 'chart.js/auto';

    type BarChartData = {
        labels: string[];
        values: number[];
        label: string;
    };

    let { data = {
        labels: ['Red', 'Blue', 'Yellow', 'Green', 'Purple', 'Orange'],
        values: [12, 19, 3, 5, 2, 3],
        label: 'Product Sales'
    } } = $props<{ data?: BarChartData }>();

    let canvas: HTMLCanvasElement;
    let chart = $state<Chart | null>(null);

    $effect(() => {
        if (!canvas) return;

        const chartData = {
            labels: data.labels,
            datasets: [{
                label: data.label,
                data: data.values,
                backgroundColor: [
                    'rgba(255, 99, 132, 0.5)',
                    'rgba(54, 162, 235, 0.5)',
                    'rgba(255, 206, 86, 0.5)',
                    'rgba(75, 192, 192, 0.5)',
                    'rgba(153, 102, 255, 0.5)',
                    'rgba(255, 159, 64, 0.5)'
                ],
                borderColor: [
                    'rgba(255, 99, 132, 1)',
                    'rgba(54, 162, 235, 1)',
                    'rgba(255, 206, 86, 1)',
                    'rgba(75, 192, 192, 1)',
                    'rgba(153, 102, 255, 1)',
                    'rgba(255, 159, 64, 1)'
                ],
                borderWidth: 1
            }]
        };

        chart = new Chart(canvas, {
            type: 'bar',
            data: chartData,
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: data.label
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true
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