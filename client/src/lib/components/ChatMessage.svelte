<script lang="ts">
    import { cn } from '$lib/utils';
    import Copy from '$lib/components/ui/icons/Copy.svelte';
    import RotateCcw from '$lib/components/ui/icons/RotateCcw.svelte';
    import Bot from '$lib/components/ui/icons/Bot.svelte';
    import User from '$lib/components/ui/icons/User.svelte';
    import { Button } from '$lib/components/ui/button';
    import { BarChart, LineChart, PieChart, ScatterPlot } from '$lib/components/charts';
    import { ChartCard } from '$lib/components/dashboard';
    // import { Toaster } from '$lib/components/ui/sonner';
    import { toast } from '$lib/hooks/use-toast';

    type ChatMessageProps = {
        content: string;
        isAi?: boolean;
        onRegenerate?: () => void;
    }

    let { content, isAi, onRegenerate }: ChatMessageProps = $props();

    // Parse content if it's a JSON string and from AI
    let steps = $state<Record<string, any>>({});
    
    // Track visible steps and processing state
    let visibleSteps = $state<string[]>([]);
    let isProcessing = $state(false);
    let currentContent = $state<string>('');
    
    $effect(() => {
        // Only process if content has changed
        if (isAi && typeof content === 'string' && content !== currentContent && !isProcessing) {
            try {
                currentContent = content;
                steps = JSON.parse(content);
                // Reset visible steps when content changes
                visibleSteps = [];
                // Start revealing steps
                revealStepsSequentially();
            } catch (e) {
                console.error('Failed to parse content:', e);
                steps = {};
            }
        }
    });

    async function revealStepsSequentially() {
        if (isProcessing) return;
        
        isProcessing = true;
        const stepIds = Object.keys(steps);
        
        for (const stepId of stepIds) {
            if (!visibleSteps.includes(stepId)) {
                await new Promise(resolve => setTimeout(resolve, 1000));
                visibleSteps = [...visibleSteps, stepId];
            }
        }
        
        isProcessing = false;
    }

    const copyToClipboard = () => {
        navigator.clipboard.writeText(content);
        toast({
            title: "Success!",
            description: "Copied to clipboard",
        });
    }

    function renderChart(chartData: any) {
        if (!chartData || !chartData.datasets) return null;
        
        // Determine chart type based on the data structure
        // For this example, we'll use BarChart as default
        return {
            component: BarChart,
            data: {
                labels: chartData.labels,
                values: chartData.datasets[0].data,
                label: chartData.datasets[0].label,
                backgroundColor: chartData.datasets[0].backgroundColor,
                xAxisLabel: "Categories",
                yAxisLabel: "Count"
            }
        };
    }

    function formatStats(stats: Record<string, any>) {
        return Object.entries(stats).map(([key, value]) => {
            if (typeof value === 'object') {
                return `${key}: ${JSON.stringify(value)}`;
            }
            return `${key}: ${value}`;
        });
    }
</script>

<style>
    .typewriter {
        overflow: hidden;
        white-space: pre-wrap;
        animation: typing 3s steps(100, end);
    }

    .simple-message {
        white-space: pre-wrap;
    }

    @keyframes typing {
        from { 
            max-height: 0;
            opacity: 0;
        }
        to { 
            max-height: 1000px;
            opacity: 1;
        }
    }
</style>

<div class={cn(
    "flex gap-4 p-6 rounded-2xl",
    isAi ? "bg-secondary/50" : "chat-gradient"
)}>
    <div
        class={cn(
            "w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0",
            isAi ? "bg-primary text-primary-foreground" : "bg-background text-foreground"
        )}
    >
        {#if isAi}
            <Bot class="w-4 h-4" />
        {:else}
            <User class="w-4 h-4" />
        {/if}
    </div>
    <div class="flex-1 space-y-4">
        {#if !isAi}
            <p class="text-sm leading-relaxed simple-message">{content}</p>
        {:else if Object.keys(steps).length === 0}
            <p class="text-sm leading-relaxed typewriter">{content}</p>
        {:else}
            <div class="space-y-6">
                {#each Object.entries(steps) as [stepId, step]}
                    {#if visibleSteps.includes(stepId)}
                        <div class="border rounded-lg p-4 space-y-3">
                            <h3 class="font-medium typewriter">Step {stepId}</h3>
                            
                            {#if step.metadata?.description}
                                <p class="text-sm text-gray-600 typewriter">{step.metadata.description}</p>
                            {/if}

                            {#if step.step_result?.stats}
                                <div class="bg-gray-50 rounded p-3">
                                    <h4 class="text-sm font-medium mb-2 typewriter">Statistics</h4>
                                    <ul class="text-sm space-y-1">
                                        {#each formatStats(step.step_result.stats) as stat}
                                            <li class="typewriter">{stat}</li>
                                        {/each}
                                    </ul>
                                </div>
                            {/if}

                            {#if step.step_result?.chart_data}
                                {@const chartInfo = renderChart(step.step_result.chart_data)}
                                {#if chartInfo}
                                    <ChartCard 
                                        title={step.step_result.chart_data.datasets[0].label || `Analysis Step ${stepId}`}
                                        class="h-[300px]"
                                    >
                                        <div class="h-full p-4">
                                            <chartInfo.component 
                                                data={chartInfo.data}
                                                class="bg-gray-800 rounded-xl h-full"
                                            />
                                        </div>
                                    </ChartCard>
                                {/if}
                            {/if}
                        </div>
                    {/if}
                {/each}
            </div>
        {/if}

        {#if isAi}
            <div class="flex gap-2">
                <Button
                    variant="secondary"
                    size="sm"
                    class="h-8"
                    onclick={copyToClipboard}
                >
                    <Copy class="w-4 h-4 mr-2" />
                    Copy
                </Button>
                {#if onRegenerate}
                    <Button
                        variant="secondary"
                        size="sm"
                        class="h-8"
                        onclick={onRegenerate}
                    >
                        <RotateCcw class="w-4 h-4 mr-2" />
                        Regenerate
                    </Button>
                {/if}
            </div>
        {/if}
    </div>
</div>