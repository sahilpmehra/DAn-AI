<script lang="ts">
    import { sessionId, isFileUploaded, tableDataStore, statsDataStore, analysisDataStore } from '$lib/stores/stores';
    import { Button } from '$lib/components/ui/button/index';
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

            isFileUploaded.set(true);
            navigate('data-summary');
        } catch (err) {
            error = err instanceof Error ? err.message : 'Upload failed';
            // toast.error(error);
        } finally {
            uploading = false;
        }
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
{/if}