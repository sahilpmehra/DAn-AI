<script lang="ts">
    import { sessionId, isFileUploaded, tableDataStore, statsDataStore, analysisDataStore } from '$lib/stores/stores';
    import { Button } from '$lib/components/ui/button/index';
    import DataTable from '$lib/components/DataTable.svelte';
    import StatsSummary from '$lib/StatsSummary.svelte';
    import AnalysisSummary from '$lib/AnalysisSummary.svelte';
    import Upload from '$lib/components/ui/icons/Upload.svelte';
    import { toast } from '$lib/hooks/use-toast';
    
    let file = $state<File | null>(null);
    let uploading = $state(false);
    let error = $state<string | null>(null);
    let isDragging = $state(false);
    let fileInput = $state<HTMLInputElement | null>(null);

    // Accept navigate function as a prop
    let { navigate } = $props<{
        navigate: (route: string) => void;
    }>();

    function handleDragOver(e: DragEvent) {
        e.preventDefault();
        isDragging = true;
    }

    function handleDragLeave() {
        isDragging = false;
    }

    function handleDrop(e: DragEvent) {
        e.preventDefault();
        isDragging = false;
        const droppedFile = e.dataTransfer?.files[0];
        if (droppedFile) handleFile(droppedFile);
    }

    function handleFileInput() {
        const selectedFile = fileInput?.files?.[0];
        if (selectedFile) handleFile(selectedFile);
    }

    async function handleFile(uploadedFile: File) {
        // Validate file type
        if (!uploadedFile.name.match(/\.(csv|xlsx)$/)) {
            // toast.error("Please upload a CSV or Excel file");
            return;
        }

        file = uploadedFile;
        uploading = true;
        error = null;

        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch('http://localhost:8000/api/v1/upload/', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error('Upload failed');
            const data = await response.json();
            sessionId.set(data.session_id);
            toast({
                title: "Success!",
                description: "File uploaded successfully!",
            });
            
            // Update state with received data
            // tableDataStore.set(data.preview_data);
            // statsDataStore.set(data.summary_stats);
            // analysisDataStore.set(data.analysis_summary);
            // Sample data for demonstration
            // Update stores instead of local state
            tableDataStore.set({
                headers: ["Age", "Income", "Education", "Job_Satisfaction"],
                data: [
                    { Age: 25, Income: 45000, Education: "Bachelor's", Job_Satisfaction: 7 },
                    { Age: 35, Income: 65000, Education: "Master's", Job_Satisfaction: 8 },
                    { Age: 28, Income: 52000, Education: "Bachelor's", Job_Satisfaction: 6 },
                    { Age: 42, Income: 85000, Education: "PhD", Job_Satisfaction: 9 },
                    { Age: 31, Income: 58000, Education: "Master's", Job_Satisfaction: 7 },
                    { Age: 25, Income: 45000, Education: "Bachelor's", Job_Satisfaction: 7 },
                    { Age: 35, Income: 65000, Education: "Master's", Job_Satisfaction: 8 },
                    { Age: 28, Income: 52000, Education: "Bachelor's", Job_Satisfaction: 6 },
                    { Age: 42, Income: 85000, Education: "PhD", Job_Satisfaction: 9 },
                    { Age: 31, Income: 58000, Education: "Master's", Job_Satisfaction: 7 },
                    { Age: 25, Income: 45000, Education: "Bachelor's", Job_Satisfaction: 7 },
                    { Age: 35, Income: 65000, Education: "Master's", Job_Satisfaction: 8 },
                    { Age: 28, Income: 52000, Education: "Bachelor's", Job_Satisfaction: 6 },
                    { Age: 42, Income: 85000, Education: "PhD", Job_Satisfaction: 9 },
                    { Age: 31, Income: 58000, Education: "Master's", Job_Satisfaction: 7 },
                ]
            });

            statsDataStore.set({
                headers: ["Metric", "Age", "Income", "Job_Satisfaction"],
                data: [
                    { Metric: "Count", Age: 5, Income: 5, Job_Satisfaction: 5 },
                    { Metric: "Mean", Age: 32.2, Income: 61000, Job_Satisfaction: 7.4 },
                    { Metric: "Std", Age: 6.5, Income: 15166, Job_Satisfaction: 1.14 },
                    { Metric: "Min", Age: 25, Income: 45000, Job_Satisfaction: 6 },
                    { Metric: "Max", Age: 42, Income: 85000, Job_Satisfaction: 9 },
                ]
            });

            analysisDataStore.set({
                summary: "The dataset contains information about employees including their age, income, education level, and job satisfaction. The data shows a positive correlation between education level and income, with PhD holders earning the highest salaries.",
                keyVariables: ["Income", "Education", "Job_Satisfaction"],
                problematicVariables: ["Department", "Manager_ID"]
            });

            isFileUploaded.set(true);
        } catch (err) {
            error = err instanceof Error ? err.message : 'Upload failed';
            // toast.error(error);
        } finally {
            uploading = false;
        }
    }

    // Function to handle navigation after successful file upload
    function handleNavigateToChat() {
        navigate('chat');
    }

    function handleNavigateToVisualizations() {
        navigate('visualizations');
    }
</script>

<h1 class="text-3xl font-bold text-center my-8">AI Data Analyst</h1>
{#if !$isFileUploaded}
    <div
        role="button"
        tabindex="0"
        aria-label="Drop your files here"
        ondragover={handleDragOver}
        ondragleave={handleDragLeave}
        ondrop={handleDrop}
        class="border-2 border-dashed rounded-lg p-12 text-center transition-colors {
            isDragging ? 'border-blue-500 bg-blue-50' : 'border-gray-300'
        }"
    >
        <Upload class="mx-auto h-12 w-12 text-gray-400" />
        <h3 class="mt-4 text-lg font-semibold">Upload your dataset</h3>
        <p class="mt-2 text-sm text-gray-500">Drag and drop your CSV or Excel file here</p>
        
        <div class="mt-4">
            <label for="file-upload" class="cursor-pointer">
                <Button variant="outline" onclick={() => fileInput && fileInput.click()}>
                    Select File
                </Button>
                <input
                    bind:this={fileInput}
                    id="file-upload"
                    type="file"
                    class="hidden"
                    accept=".csv,.xlsx,.xls"
                    onchange={handleFileInput}
                />
            </label>
        </div>

        {#if uploading}
            <div class="text-blue-600 mt-4">Uploading...</div>
        {/if}

        {#if file && !uploading && !error}
            <div class="text-green-600 mt-4">File selected: {file.name}</div>
        {/if}
    </div>
{:else}
    <div class="space-y-8">
        {#if $tableDataStore.headers.length > 0}
            <div class="bg-white rounded-lg shadow">
                <DataTable data={$tableDataStore.data} headers={$tableDataStore.headers} />
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
                <button
                    class="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
                    onclick={handleNavigateToVisualizations}
                >
                    Create Visualizations for Key Variables
                </button>
                <button
                    class="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
                    onclick={handleNavigateToChat}
                >
                    Chat with Dan AI
                </button>
            </div>
        {/if}
    </div>
{/if}