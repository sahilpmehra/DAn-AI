<script lang="ts">
    import { tableDataStore, statsDataStore, analysisDataStore, sessionId } from '$lib/stores/stores';
    import { Button } from '$lib/components/ui/button/index';
    import DataSample from '$lib/DataSample.svelte';
    import StatsSummary from '$lib/StatsSummary.svelte';
    import AnalysisSummary from '$lib/AnalysisSummary.svelte';

    // Accept navigate function as a prop
    let { navigate } = $props<{
        navigate: (route: string) => void;
    }>();

    // Add loading state
    let loading = $state(false);

    // Sample data for demonstration
    // Update stores instead of local state
    // tableDataStore.set({
    //     headers: ["Age", "Income", "Education", "Job_Satisfaction"],
    //     data: [
    //         { Age: 25, Income: 45000, Education: "Bachelor's", Job_Satisfaction: 7 },
    //         { Age: 35, Income: 65000, Education: "Master's", Job_Satisfaction: 8 },
    //         { Age: 28, Income: 52000, Education: "Bachelor's", Job_Satisfaction: 6 },
    //         { Age: 42, Income: 85000, Education: "PhD", Job_Satisfaction: 9 },
    //         { Age: 31, Income: 58000, Education: "Master's", Job_Satisfaction: 7 },
    //         { Age: 25, Income: 45000, Education: "Bachelor's", Job_Satisfaction: 7 },
    //         { Age: 35, Income: 65000, Education: "Master's", Job_Satisfaction: 8 },
    //         { Age: 28, Income: 52000, Education: "Bachelor's", Job_Satisfaction: 6 },
    //         { Age: 42, Income: 85000, Education: "PhD", Job_Satisfaction: 9 },
    //         { Age: 31, Income: 58000, Education: "Master's", Job_Satisfaction: 7 },
    //         { Age: 25, Income: 45000, Education: "Bachelor's", Job_Satisfaction: 7 },
    //         { Age: 35, Income: 65000, Education: "Master's", Job_Satisfaction: 8 },
    //         { Age: 28, Income: 52000, Education: "Bachelor's", Job_Satisfaction: 6 },
    //         { Age: 42, Income: 85000, Education: "PhD", Job_Satisfaction: 9 },
    //         { Age: 31, Income: 58000, Education: "Master's", Job_Satisfaction: 7 },
    //     ]
    // });
    
    // statsDataStore.set({
    //     headers: ["Metric", "Age", "Income", "Job_Satisfaction"],
    //     data: [
    //         { Metric: "Count", Age: 5, Income: 5, Job_Satisfaction: 5 },
    //         { Metric: "Mean", Age: 32.2, Income: 61000, Job_Satisfaction: 7.4 },
    //         { Metric: "Std", Age: 6.5, Income: 15166, Job_Satisfaction: 1.14 },
    //         { Metric: "Min", Age: 25, Income: 45000, Job_Satisfaction: 6 },
    //         { Metric: "Max", Age: 42, Income: 85000, Job_Satisfaction: 9 },
    //     ]
    // });

    // analysisDataStore.set({
    //     summary: "The dataset contains information about employees including their age, income, education level, and job satisfaction. The data shows a positive correlation between education level and income, with PhD holders earning the highest salaries.",
    //     keyVariables: ["Income", "Education", "Job_Satisfaction"],
    //     problematicVariables: ["Department", "Manager_ID"]
    // });

    // Add function to fetch data from backend
    async function fetchDataFromBackend() {
        loading = true; // Set loading to true when fetching starts
        try {
            const response = await fetch('http://localhost:8000/api/v1/data-summary/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ session_id: $sessionId }),
            });
            const data = await response.json();
            
            // Update stores with received data
            tableDataStore.set(data.preview_data);
            statsDataStore.set(data.summary_stats);
            analysisDataStore.set(data.analysis_summary);
        } catch (error) {
            console.error('Failed to fetch data:', error);
            // Handle error appropriately
        } finally {
            loading = false; // Set loading to false when fetching is done
        }
    }

    // Check if stores are empty and fetch data if needed
    $effect(() => {
        const hasTableData = $tableDataStore.headers.length > 0;
        const hasStatsData = $statsDataStore.headers.length > 0;
        const hasAnalysisData = Boolean($analysisDataStore.summary);

        if (!hasTableData || !hasStatsData || !hasAnalysisData) {
            fetchDataFromBackend();
        }
    });

    // Function to handle navigation after successful display of data summary
    function handleNavigateToChat() {
        navigate('chat');
    }

    function handleNavigateToVisualizations() {
        navigate('visualizations');
    }
</script>

<h1 class="text-3xl font-bold text-center my-8">AI Data Analyst</h1>
<div class="space-y-8">
    {#if loading}
        <div class="text-center">Loading data, please wait...</div>
    {:else}
        {#if $tableDataStore.headers.length > 0}
            <div class="bg-white rounded-lg shadow">
                <DataSample data={$tableDataStore.data} headers={$tableDataStore.headers} />
            </div>
        {/if}

        {#if $statsDataStore.headers.length > 0}
            <div class="bg-white rounded-lg shadow">
                <StatsSummary data={$statsDataStore.data} headers={$statsDataStore.headers} />
            </div>
        {/if}

        {#if $analysisDataStore.summary}
            <div class="bg-white rounded-lg shadow">
                <AnalysisSummary 
                    keyVariables={$analysisDataStore.keyVariables} 
                    problematicVariables={$analysisDataStore.problematicVariables} 
                    summary={$analysisDataStore.summary}
                />
            </div>
        {/if}
        {#if $tableDataStore.headers.length > 0 && $statsDataStore.headers.length > 0 && $analysisDataStore.summary}
            <!-- Add navigation buttons after successful file upload -->
            <div class="flex justify-center gap-4 pb-6">
                <Button
                    class="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
                    onclick={handleNavigateToVisualizations}
                >
                    Create Visualizations for Key Variables
                </Button>
                <Button
                    class="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
                    onclick={handleNavigateToChat}
                >
                    Chat with Dan AI
                </Button>
            </div>
        {/if}
    {/if}
</div>