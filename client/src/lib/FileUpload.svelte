<script lang="ts">
    import { sessionId } from './stores';
    
    let file: File | null = $state(null);
    let uploading = $state(false);
    let error = $state<string | null>(null);
  
    async function handleFileUpload(event: Event) {
      const input = event.target as HTMLInputElement;
      if (!input.files?.length) return;
      
      file = input.files[0];
      
      // Validate file type
      if (!file.name.match(/\.(csv|xlsx)$/)) {
        error = "Please upload a CSV or Excel file";
        file = null;
        return;
      }
  
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
        
        // Emit success event or update global state
      } catch (err) {
        error = err instanceof Error ? err.message : 'Upload failed';
      } finally {
        uploading = false;
      }
    }
  </script>
  
  <div class="w-full max-w-4xl mx-auto p-4">
    <label class="block mb-4">
      <span class="text-gray-700">Upload Dataset (CSV or Excel)</span>
      <input
        type="file"
        accept=".csv,.xlsx"
        onchange={handleFileUpload}
        class="mt-1 block w-full text-sm text-gray-500
          file:mr-4 file:py-2 file:px-4
          file:rounded-full file:border-0
          file:text-sm file:font-semibold
          file:bg-violet-50 file:text-violet-700
          hover:file:bg-violet-100"
      />
    </label>
  
    {#if uploading}
      <div class="text-blue-600">Uploading...</div>
    {/if}
  
    {#if error}
      <div class="text-red-600">{error}</div>
    {/if}
  
    {#if file && !uploading && !error}
      <div class="text-green-600">File selected: {file.name}</div>
    {/if}
  </div>