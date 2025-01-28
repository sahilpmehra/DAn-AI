<script lang="ts">
    import Chart from 'chart.js/auto';

    type ScatterPoint = {
        x: number;
        y: number;
    };

    type ScatterChartData = {
        points: ScatterPoint[];
        label: string;
        xAxisLabel: string;
        yAxisLabel: string;
    };

    let { data = {
        points: [
            { x: 25, y: 30000 },
            { x: 30, y: 45000 },
            { x: 35, y: 55000 },
            { x: 40, y: 70000 },
            { x: 45, y: 85000 },
            { x: 50, y: 95000 },
            { x: 55, y: 100000 },
        ],
        label: 'Age vs. Income Correlation',
        xAxisLabel: 'Age',
        yAxisLabel: 'Income ($)'
    } } = $props<{ data?: ScatterChartData }>();

    let canvas: HTMLCanvasElement;
    let chart = $state<Chart | null>(null);

    $effect(() => {
        if (!canvas) return;

        const chartData = {
            datasets: [{
                label: data.label,
                data: data.points,
                backgroundColor: 'rgba(75, 192, 192, 0.5)',
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 1
            }]
        };

        chart = new Chart(canvas, {
            type: 'scatter',
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
                    x: {
                        title: {
                            display: true,
                            text: data.xAxisLabel
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: data.yAxisLabel
                        }
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