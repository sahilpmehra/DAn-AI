<script lang="ts">
    import Chart from 'chart.js/auto';
    import { untrack } from 'svelte';

    type BarChartData = {
        labels: string[];
        values: number[];
        label: string;
        xAxisLabel?: string;
        yAxisLabel?: string;
    };

    let { data = {
        labels: ['Red', 'Blue', 'Yellow', 'Green', 'Purple', 'Orange'],
        values: [12, 19, 3, 5, 2, 3],
        label: 'Product Sales'
    }, class: className = '' } = $props<{ data?: BarChartData, class?: string }>();

    let canvas: HTMLCanvasElement;
    let chart: Chart | null = null;

    $effect(() => {
        if (!canvas) return;

        untrack(() => {
            if (chart) chart.destroy();
            chart = new Chart(canvas, {
                type: 'bar',
                data: {
                    labels: [...data.labels],
                    datasets: [{
                        label: data.label,
                        data: [...data.values],
                        backgroundColor: [
                            'rgba(255, 99, 132, 0.5)',
                            'rgba(54, 162, 235, 0.5)',
                            'rgba(255, 206, 86, 0.5)',
                            'rgba(75, 192, 192, 0.5)',
                            'rgba(153, 102, 255, 0.5)',
                            'rgba(255, 159, 64, 0.5)'
                        ],
                        borderWidth: 1,
                        borderRadius: 10
                    }]
                },
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
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: data.yAxisLabel
                            }
                        }
                    },
                },
            });
        });

        return () => {
            chart?.destroy();
        };
    });
</script>

<canvas bind:this={canvas} class={className}></canvas> 