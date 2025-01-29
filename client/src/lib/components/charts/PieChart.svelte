<script lang="ts">
    import Chart from 'chart.js/auto';

    type PieChartData = {
        labels: string[];
        values: number[];
        title: string;
    };

    let { data = {
        labels: ['Desktop', 'Mobile', 'Tablet'],
        values: [300, 500, 200],
        title: 'Device Usage Distribution'
    }, class: className = '' } = $props<{ data?: PieChartData, class?: string }>();

    let canvas: HTMLCanvasElement;
    let chart = $state<Chart | null>(null);

    $effect(() => {
        if (!canvas) return;

        const chartData = {
            labels: data.labels,
            datasets: [{
                data: data.values,
                backgroundColor: [
                    'rgba(255, 99, 132, 1)',
                    'rgba(54, 162, 235, 1)',
                    'rgba(255, 206, 86, 1)'
                ],
                borderColor: [
                    'rgba(255, 99, 132, 1)',
                    'rgba(54, 162, 235, 1)',
                    'rgba(255, 206, 86, 1)'
                ],
                borderWidth: 1
            }]
        };

        chart = new Chart(canvas, {
            type: 'doughnut',
            data: chartData,
            options: {
                plugins: {
                    title: {
                        display: true,
                        text: data.title
                    }
                }
            }
        });

        return () => {
            chart?.destroy();
        };
    });
</script>

<canvas bind:this={canvas} class={className}></canvas> 