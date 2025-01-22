<script lang="ts">
    import { Button } from "$lib/components/ui/button";
    import { Select, SelectItem, SelectTrigger, SelectContent } from "$lib/components/ui/select";
    import ChevronLeft from "$lib/components/ui/icons/ChevronLeft.svelte";
    import ChevronRight from "$lib/components/ui/icons/ChevronRight.svelte";

    let isCollapsed = $state(false);
    
    // Define options
    const variables = [
        { value: "var1", label: "Variable 1" },
        { value: "var2", label: "Variable 2" },
        { value: "var3", label: "Variable 3" }
    ];
    
    const chartTypes = [
        { value: "bar", label: "Bar Chart" },
        { value: "line", label: "Line Chart" },
        { value: "pie", label: "Pie Chart" }
    ];
    
    // State
    let selectedVariable = $state("");
    let selectedChart = $state("");
    
    // Derived values for triggers
    let variableTriggerContent = $derived(
        variables.find(v => v.value === selectedVariable)?.label ?? "Choose variable"
    );
    let chartTriggerContent = $derived(
        chartTypes.find(c => c.value === selectedChart)?.label ?? "Select chart"
    );
</script>

<div
    class={`bg-white border-l transition-all duration-300 ${
        isCollapsed ? "w-12" : "w-64"
    }`}
>
    <Button
        variant="ghost"
        size="icon"
        class="w-12 h-12"
        onclick={() => isCollapsed = !isCollapsed}
    >
        {#if isCollapsed}
            <ChevronLeft />
        {:else}
            <ChevronRight />
        {/if}
    </Button>
    {#if !isCollapsed}
        <div class="p-4 space-y-4">
            <div class="space-y-2">
                <label for="select-variable" class="text-sm font-medium">Select Variable</label>
                <Select id="select-variable" type="single" bind:value={selectedVariable}>
                    <SelectTrigger class="w-full">
                        {variableTriggerContent}
                    </SelectTrigger>
                    <SelectContent>
                        {#each variables as variable}
                            <SelectItem value={variable.value} label={variable.label}>
                                {variable.label}
                            </SelectItem>
                        {/each}
                    </SelectContent>
                </Select>
            </div>
            <div class="space-y-2">
                <label for="select-chart" class="text-sm font-medium">Chart Type</label>
                <Select id="select-chart" type="single" bind:value={selectedChart}>
                    <SelectTrigger class="w-full">
                        {chartTriggerContent}
                    </SelectTrigger>
                    <SelectContent>
                        {#each chartTypes as chartType}
                            <SelectItem value={chartType.value} label={chartType.label}>
                                {chartType.label}
                            </SelectItem>
                        {/each}
                    </SelectContent>
                </Select>
            </div>
            <Button class="w-full">Apply Filters</Button>
        </div>
    {/if}
</div>